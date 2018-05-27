(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  /** Trainer(net, options) - 训练器 构造函数
   * net - 神经网络定义(是简化版)
   * options - 设定一些基础的计算参数
   * 构造函数同时需要初始化计数器(this.k 即epoch#), 梯度累计(gsum[]), 加速度累计(xsum[])
   */
  var Trainer = function(net, options) {

    this.net = net;

    var options = options || {};
    this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
    this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
    this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
    this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
    this.method = typeof options.method !== 'undefined' ? options.method : 'sgd'; // sgd/adagrad/adadelta/windowgrad/netsterov

    this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;
    this.ro = typeof options.ro !== 'undefined' ? options.ro : 0.95; // used in adadelta
    this.eps = typeof options.eps !== 'undefined' ? options.eps : 1e-6; // used in adadelta

    this.k = 0; // iteration counter
    this.gsum = []; // last iteration gradients (used for momentum calculations)
    this.xsum = []; // used in adadelta
  }

  /** train(x, y) - 训练函数
   * 
   * [训练过程]
   * 整体框架采用的是BP神经网络, 每次训练(每张图)分成 前向计算/前向计算/权重更新 阶段
   * 
   * 1. [前向计算]
   *    forward计算会填充所有的in_act/out_act权重信息
   *    参见net的forward函数
   *    
   * 2. [后向计算]
   *    backward计算会针对上一层传回的梯度计算每一个节点的梯度信息
   *    net的最后一层为softmax会接收一个参数y, 其他层则将梯度后向反馈传递下去
   *    参见net的backward函数, 最终softmax层返回loss其他层完成计算
   *    
   * 3. [权重更新]
   * 3.1 [数据结构]
   *    前2步, 已计算获得了每个层的权重和梯度, 这些信息通过每层/网络的getParamsAndGrads()函数汇总上来
   *    params存放权重, grads存放梯度, l1/2_decay_mul存放衰减系数 为了使整个网络不至于权重扩散
   * 
   * 3.2 [循环分析]
   *    构造了双层循环pglist 所有网络中的权重p和梯度g的列表, plen 每个pglist中的向量数量
   *    
   * 3.3 [调整梯度]
   *    梯度衰减通过权重的线性组合求得
   *       l1_grad = l1_decay * sign(weight)
   *       l2_grad = l2_decay * weight
   *    调整梯度
   *    g' = (l2_grad + l1_grad + g)
   *    
   * 3.4 [权重更新]
   *    权重更新有很多种方法, 正如教程中所写: 
   *    adagrad, windowgrad, adadelta, nesterov, sgd, 及vanilla sgd
   *    主要还是一种物理的方式去更快更好的逼近最优值, 计算出对应的权重调整量
   *    这些方法最基础的就是sgd(simple gradient descendent) 也是默认方法
   *    比如vanilla sgd: weight = weight - this.learning_rate * g'
   *     
   *    注意: 这里是对网络中所有权重都进行调整, 逐步的每个点的权重都会趋于最优化
   *    但是到底多少个epoch后会收敛此处并未给出, 就看样本数量了
   * */
  Trainer.prototype = {
    train: function(x, y) {

      var start = new Date().getTime();
      this.net.forward(x, true); // also set the flag that lets the net know we're just training
      var end = new Date().getTime();
      var fwd_time = end - start;

      var start = new Date().getTime();
      var cost_loss = this.net.backward(y);
      var l2_decay_loss = 0.0;
      var l1_decay_loss = 0.0;
      var end = new Date().getTime();
      var bwd_time = end - start;
      
      this.k++;
      if(this.k % this.batch_size === 0) {

        var pglist = this.net.getParamsAndGrads();

        // initialize lists for accumulators. Will only be done once on first iteration
        if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
          // only vanilla sgd doesnt need either lists
          // momentum needs gsum
          // adagrad needs gsum
          // adadelta needs gsum and xsum
          for(var i=0;i<pglist.length;i++) {
            this.gsum.push(global.zeros(pglist[i].params.length));
            if(this.method === 'adadelta') {
              this.xsum.push(global.zeros(pglist[i].params.length));
            } else {
              this.xsum.push([]); // conserve memory
            }
          }
        }

        // perform an update for all sets of weights
        for(var i=0;i<pglist.length;i++) {
          var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
          var p = pg.params;
          var g = pg.grads;

          // learning rate for some parameters.
          var l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
          var l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
          var l2_decay = this.l2_decay * l2_decay_mul;
          var l1_decay = this.l1_decay * l1_decay_mul;

          var plen = p.length;
          for(var j=0;j<plen;j++) {
            l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
            l1_decay_loss += l1_decay*Math.abs(p[j]);
            var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
            var l2grad = l2_decay * (p[j]);

            var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

            var gsumi = this.gsum[i];
            var xsumi = this.xsum[i];
            if(this.method === 'adagrad') {
              // adagrad update
              gsumi[j] = gsumi[j] + gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
              p[j] += dx;
            } else if(this.method === 'windowgrad') {
              // this is adagrad but with a moving window weighted average
              // so the gradient is not accumulated over the entire history of the run. 
              // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
              p[j] += dx;
            } else if(this.method === 'adadelta') {
              // assume adadelta if not sgd or adagrad
              gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
              var dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
              xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
              p[j] += dx;
            } else if(this.method === 'nesterov') {
            	var dx = gsumi[j];
            	gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                p[j] += dx;
            } else {
              // assume SGD
              if(this.momentum > 0.0) {
                // momentum update
                var dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                gsumi[j] = dx; // back this up for next iteration of momentum
                p[j] += dx; // apply corrected gradient
              } else {
                // vanilla sgd
                p[j] +=  - this.learning_rate * gij;
              }
            }
            g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
          }
        }
      }

      // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
      // in future, TODO: have to completely redo the way loss is done around the network as currently 
      // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
      // and it should all be computed correctly and automatically. 
      return {fwd_time: fwd_time, bwd_time: bwd_time, 
              l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
              cost_loss: cost_loss, softmax_loss: cost_loss, 
              loss: cost_loss + l1_decay_loss + l2_decay_loss}
    }
  }
  
  global.Trainer = Trainer;
  global.SGDTrainer = Trainer; // backwards compatibility
})(convnetjs);
