(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  /**
   * [概述]
   * 在每个卷积层后可能会有一个池化层(pooling layer). 
   * 这个池化层从卷积层中取得一个矩形块. 从输入小块二次抽样, 产生单独的输出. 
   * 目前有几种方法来做这样的池化, 比如取平均值或最大值或神经元区块中已学习的线性组合
   * 
   * 本例中实现的是max-pooling最大池化方法
   * 
   * [成员变量]
   *    基础变量 layer_type : 'pool'
   *    输入变量 in_sx, in_sy, in_depth, in_act
   *    滤镜变量 sx, sy
   *    输出变量 out_sx, out_sy, out_depth, out_act
   *    其他变量 stride, pad, switchx, switchy
   *           (用于存储在池化过程中找到的最值的坐标位置 switch[x, y] 覆盖了out的所有变量的来源坐标)
   *
   * 实现BP Propagation的卷积神经网络层 
   * 
   * [数据结构]
   * V/in_act  = [ in_sx  x in_sy  x in_depth ]
   * switchxy  = [ out_sx x out_sy            x out_depth ]
   * A/out_act = [ out_sx x out_sy            x out_depth ]
   * 
   * [循环分析]
   * 在池化层中forward(), backward()两个函数采用双层循环方式
   * 输出层按照输出格式循环, 然后是滤镜层/卷积层 找出滤镜平面中所对应的最大值的坐标
   * (优化点)正是由于在forward()计算中保留了这些坐标, 在backward()中就不需要这层循环, 直接进行访问
   *    输出层循环: out_depth(d) -> out_sx(ax) -> out_sy(ay) ->
   *    滤镜层循环: sx(fx) -> sy(fy)
   * 
   * [数值计算]
   * forward()计算:
   *    A<l>.w [ax, ay, d] := max( A<l-1>.w[ox, oy, d] )
   *    switch<l>[x, y, d] := corresponding(ox, oy, d)
   * 
   * backward()计算:
   *    A<l-1>.dw[ax, ay, d] += A<l>.dw[ax, ay, d]
   *    此处只修改最大值所在的那个位置的梯度
   * 
   **/
  
  
  /** PoolLayer(opt) - 池化层 构造函数
   * 下面主要为其构造函数用于初始化该卷积层
   * opt 参数说明
   *    输入层参数 in_sx, in_sy, in_depth
   *    滤镜参数 sx, sy
   *    计算参数 pad - 边缘填充, stride - 步长
   * */
  var PoolLayer = function(opt) {

    var opt = opt || {};

    // required
    this.sx = opt.sx; // filter size
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 2;
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

    // computed
    this.out_depth = this.in_depth;
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'pool';
    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
  }

  PoolLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;

      var A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
      
      var n=0; // a counter for switches
      for(var d=0;d<this.out_depth;d++) {
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            // convolve centered at this particular location
            var a = -99999; // hopefully small enough ;\
            var winx=-1,winy=-1;
            for(var fx=0;fx<this.sx;fx++) {
              for(var fy=0;fy<this.sy;fy++) {
                var oy = y+fy;
                var ox = x+fx;
                if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                  var v = V.get(ox, oy, d);
                  // perform max pooling and store pointers to where
                  // the max came from. This will speed up backprop 
                  // and can help make nice visualizations in future
                  if(v > a) { a = v; winx=ox; winy=oy;}
                }
              }
            }
            this.switchx[n] = winx;
            this.switchy[n] = winy;
            n++;
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() { 
      // pooling layers have no parameters, so simply compute 
      // gradient wrt data here
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      var A = this.out_act; // computed in forward pass 

      var n = 0;
      for(var d=0;d<this.out_depth;d++) {
        var x = -this.pad;
        var y = -this.pad;
        for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
          y = -this.pad;
          for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {

            var chain_grad = this.out_act.get_grad(ax,ay,d);
            V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
            n++;

          }
        }
      }
    },
    getParamsAndGrads: function() {
      return [];
    },
    toJSON: function() {
      var json = {};
      json.sx = this.sx;
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth;
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.pad = this.pad;
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx;
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth;
      this.pad = typeof json.pad !== 'undefined' ? json.pad : 0; // backwards compatibility
      this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth); // need to re-init these appropriately
      this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);
    }
  }

  global.PoolLayer = PoolLayer;

})(convnetjs);
