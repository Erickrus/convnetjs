(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  var assert = global.assert;

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  var Net = function(options) {
    this.layers = [];
  }

  /** 
   * [概述]
   * 一个Net网络就是由很多层不同类型的神经元组成的整体
   * Net作为一个除了组织数据还形成了BP框架, 通过"模板模式"调用不同层的BP算法
   * 
   * [使用方法]
   * 定义对象, makelayers, 最后用trainer对象训练这个网络
   * 
   *  */
  Net.prototype = {
    
	/** makeLayers(defs) - 用于定义一个网络
	 * 接受defs/opt传递进来的网络定义信息
	 * 在这里def的结构其实就是具体网络构造函数的opt选项对象
	 * 系统通过甄别每组不同的输入自动建立层与层之间的关系
	 * 
	 * 值得注意的是: 为了减少输入以及错误, 补充一些默认值
	 * 通过内嵌函数desugar()方法增加了一些必要的activation层
	 * 
	 * 操作规则:
	 * 1) 当前位置前插入: [fc] => softmax/regression/svm
	 * 2) 当前位置后插入: 任何含activation操作 => [relu/sigmoid/tanh/maxout]
	 *    
	 * 1) 事实上, softmax/regression/svm等均为分类, 这些过程之前必须有一个全连接层, 把卷积信息隔离开
	 * 2) 在层后面插入是为了简化层的定义, 程序非常灵活可以定义不同的激活层函数
	 **/

    // takes a list of layer definitions and creates the network layer objects
    makeLayers: function(defs) {

      // few checks
      assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
      assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

      // desugar layer_defs for adding activation, dropout layers etc
      var desugar = function() {
        var new_defs = [];
        for(var i=0;i<defs.length;i++) {
          var def = defs[i];
          
          if(def.type==='softmax' || def.type==='svm') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_classes});
          }

          if(def.type==='regression') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_neurons});
          }

          if((def.type==='fc' || def.type==='conv') 
              && typeof(def.bias_pref) === 'undefined'){
            def.bias_pref = 0.0;
            if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
              def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
              // otherwise it's technically possible that a relu unit will never turn on (by chance)
              // and will never get any gradient and never contribute any computation. Dead relu.
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
            else if (def.activation==='tanh') { new_defs.push({type:'tanh'}); }
            else if (def.activation==='maxout') {
              // create maxout activation, and pass along group size, if provided
              var gs = def.group_size !== 'undefined' ? def.group_size : 2;
              new_defs.push({type:'maxout', group_size:gs});
            }
            else { console.log('ERROR unsupported activation ' + def.activation); }
          }
          if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
            new_defs.push({type:'dropout', drop_prob: def.drop_prob});
          }

        }
        return new_defs;
      }
      defs = desugar(defs);

      // create the layers
      this.layers = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        if(i>0) {
          var prev = this.layers[i-1];
          def.in_sx = prev.out_sx;
          def.in_sy = prev.out_sy;
          def.in_depth = prev.out_depth;
        }

        switch(def.type) {
          case 'fc': this.layers.push(new global.FullyConnLayer(def)); break;
          case 'lrn': this.layers.push(new global.LocalResponseNormalizationLayer(def)); break;
          case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
          case 'input': this.layers.push(new global.InputLayer(def)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
          case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
          case 'conv': this.layers.push(new global.ConvLayer(def)); break;
          case 'pool': this.layers.push(new global.PoolLayer(def)); break;
          case 'relu': this.layers.push(new global.ReluLayer(def)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
          case 'tanh': this.layers.push(new global.TanhLayer(def)); break;
          case 'maxout': this.layers.push(new global.MaxoutLayer(def)); break;
          case 'svm': this.layers.push(new global.SVMLayer(def)); break;
          default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
        }
      }
    },

    /** forward(V, training) - 计算前向信息
     * 关于training状态, 在trainer中调用的时候, 会传入true
     * 如从其他地方调用, 或是省去training这参数则默认为预测模式 */
    // forward prop the network. 
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    forward: function(V, is_training) {
      if(typeof(is_training) === 'undefined') is_training = false;
      var act = this.layers[0].forward(V, is_training);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
      }
      return act;
    },

    /** getCostLoss(V, y) - 获取最后一层 分类器的loss作为结果
     * 只有最后一层的backward()才会提供loss值 */
    getCostLoss: function(V, y) {
      this.forward(V, false);
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y);
      return loss;
    },
    
    /** backward(y) - 反向调用所有层的backward()函数, 返回loss结果 */
    // backprop: compute gradients wrt all parameters
    backward: function(y) {
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
      for(var i=N-2;i>=0;i--) { // first layer assumed input
        this.layers[i].backward();
      }
      return loss;
    },
    /** getParamsAndGrads() - 返回整个网络的参数和梯度信息 */
    getParamsAndGrads: function() {
      // accumulate parameters and gradients for the entire network
      var response = [];
      for(var i=0;i<this.layers.length;i++) {
        var layer_reponse = this.layers[i].getParamsAndGrads();
        for(var j=0;j<layer_reponse.length;j++) {
          response.push(layer_reponse[j]);
        }
      }
      return response;
    },
    /** getPrediction() - 获取预测信息
     * 简化argmax过程, 当计算完成后, 在softmax层最终结果会有一个概率权重信息
     * 函数比对权重信息， 获取最大的一个id及其概率信息
     * 
     * 系统将默认最后一层为softmax, 否则报错
     * */
    getPrediction: function() {
      // this is a convenience function for returning the argmax
      // prediction, assuming the last layer of the net is a softmax
      var S = this.layers[this.layers.length-1];
      assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

      var p = S.out_act.w;
      var maxv = p[0];
      var maxi = 0;
      for(var i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i;}
      }
      return maxi; // return index of the class with highest class probability
    },
    toJSON: function() {
      var json = {};
      json.layers = [];
      for(var i=0;i<this.layers.length;i++) {
        json.layers.push(this.layers[i].toJSON());
      }
      return json;
    },
    fromJSON: function(json) {
      this.layers = [];
      for(var i=0;i<json.layers.length;i++) {
        var Lj = json.layers[i]
        var t = Lj.layer_type;
        var L;
        if(t==='input') { L = new global.InputLayer(); }
        if(t==='relu') { L = new global.ReluLayer(); }
        if(t==='sigmoid') { L = new global.SigmoidLayer(); }
        if(t==='tanh') { L = new global.TanhLayer(); }
        if(t==='dropout') { L = new global.DropoutLayer(); }
        if(t==='conv') { L = new global.ConvLayer(); }
        if(t==='pool') { L = new global.PoolLayer(); }
        if(t==='lrn') { L = new global.LocalResponseNormalizationLayer(); }
        if(t==='softmax') { L = new global.SoftmaxLayer(); }
        if(t==='regression') { L = new global.RegressionLayer(); }
        if(t==='fc') { L = new global.FullyConnLayer(); }
        if(t==='maxout') { L = new global.MaxoutLayer(); }
        if(t==='svm') { L = new global.SVMLayer(); }
        L.fromJSON(Lj);
        this.layers.push(L);
      }
    }
  }
  
  global.Net = Net;
})(convnetjs);
