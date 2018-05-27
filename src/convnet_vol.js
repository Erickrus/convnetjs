(function(global) {
  "use strict";

  /** Vol(sx, sy, depth, c) - Vol对象是网络中所有数据的基础building block.
   * 实际上就是一个由宽sx高sy深depth三维组成的数组.
   * 这个结构用来保存所有的Filter, Volumes, 权重, 梯度信息等
   * 最后c是初始化常数, 可以不指定, 返回一个随机Vol */
  // Vol is the basic building block of all data in a net.
  // it is essentially just a 3D volume of numbers, with a
  // width (sx), height (sy), and depth (depth).
  // it is used to hold data for all filters, all volumes,
  // all weights, and also stores all gradients w.r.t. 
  // the data. c is optionally a value to initialize the volume
  // with. If c is missing, fills the Vol with random numbers.
  var Vol = function(sx, sy, depth, c) {
    // this is how you check if a variable is an array. Oh, Javascript :)
    if(Object.prototype.toString.call(sx) === '[object Array]') {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.sy = 1;
      this.depth = sx.length;
      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.w = global.zeros(this.depth);
      this.dw = global.zeros(this.depth);
      for(var i=0;i<this.depth;i++) {
        this.w[i] = sx[i];
      }
    } else {
      // we were given dimensions of the vol
      /** 对象主要参数及含义
       * sx - 宽度, sy - 高度, depth - 深度
       * w[sx * sy * depth] - 正则化权重
       * dw[sx * sy * depth] - 梯度 */
      this.sx = sx;
      this.sy = sy;
      this.depth = depth;
      var n = sx*sy*depth;
      this.w = global.zeros(n);
      this.dw = global.zeros(n);
      if(typeof c === 'undefined') {
        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        var scale = Math.sqrt(1.0/(sx*sy*depth));
        for(var i=0;i<n;i++) { 
          this.w[i] = global.randn(0.0, scale);
        }
      } else {
        for(var i=0;i<n;i++) { 
          this.w[i] = c;
        }
      }
    }
  }

  /** Vol 基础操作, 对象方法 */
  Vol.prototype = {
	/** Vol.get(x, y, d) - 获得所在位置的权重 */
    get: function(x, y, d) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      return this.w[ix];
    },
    /** Vol.set(x, y, d, v) - 设置所在位置的权重为v */
    set: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] = v; 
    },
    /** Vol.add(x, y, d, v) - 设置所在位置的权重+v */
    add: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] += v; 
    },
    /** Vol.get_grad(x, y, d) - 获取所在位置的梯度 */
    get_grad: function(x, y, d) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      return this.dw[ix]; 
    },
    /** Vol.set_grad(x, y, d, v) - 设置所在位置的梯度为v */
    set_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] = v; 
    },
    /** Vol.add_grad(x, y, d, v) - 设置所在位置的梯度+v */
    add_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] += v; 
    },
    /** Vol.cloneAndZero() - 返回一个维度相同,数值初始化为0的Vol对象 */
    cloneAndZero: function() { return new Vol(this.sx, this.sy, this.depth, 0.0)},
    /** Vol.clone() - 返回一个副本Vol对象 */
    clone: function() {
      var V = new Vol(this.sx, this.sy, this.depth, 0.0);
      var n = this.w.length;
      for(var i=0;i<n;i++) { V.w[i] = this.w[i]; }
      return V;
    },
    /** Vol.addFrom(V) - 矩阵加操作, 只复制权重部分 */
    addFrom: function(V) { for(var k=0;k<this.w.length;k++) { this.w[k] += V.w[k]; }},
    /** Vol.addFrom(V) - 矩阵V缩放a倍后加操作 */
    addFromScaled: function(V, a) { for(var k=0;k<this.w.length;k++) { this.w[k] += a*V.w[k]; }},
    /** Vol.setConst(a) - 将权重统一设置为常数a  */
    setConst: function(a) { for(var k=0;k<this.w.length;k++) { this.w[k] = a; }},
    /** Vol.toJSON() - 转换为json, 目前未全部完成 */
    toJSON: function() {
      // todo: we may want to only save d most significant digits to save space
      var json = {}
      json.sx = this.sx; 
      json.sy = this.sy;
      json.depth = this.depth;
      json.w = this.w;
      return json;
      // we wont back up gradients to save space
    },
    /** Vol.fromJSON() - 加载json, 此处只加载维度信息和权重，梯度初始化为0 */
    fromJSON: function(json) {
      this.sx = json.sx;
      this.sy = json.sy;
      this.depth = json.depth;

      var n = this.sx*this.sy*this.depth;
      this.w = global.zeros(n);
      this.dw = global.zeros(n);
      // copy over the elements.
      for(var i=0;i<n;i++) {
        this.w[i] = json.w[i];
      }
    }
  }

  global.Vol = Vol;
})(convnetjs);
