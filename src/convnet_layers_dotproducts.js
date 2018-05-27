(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  
  /**
   * [成员变量]
   *    基础变量 layer_type : 'conv'
   *    输入变量 in_sx, in_sy, in_depth, in_act
   *    输出变量 out_sx, out_sy (此2项由输入几何性质以及stride,pad共同决定), out_depth(滤镜数量), out_act
   *    卷积变量 stride, pad, biases (偏置项 1x1xdepth的Volume作为卷积层的输入项)
   *    滤镜变量 sx, sy(通常sx==sy滤镜为正方形), filters(一个 sx x sy x in_depth x out_depth(滤镜数) 的一个4维Tensor)
   *    其他变量 l1_decay_mul, l2_decay_mul 
   *
   * 实现BP Propagation的卷积神经网络层 
   * 
   * [计算框架]
   * 本层(l)的主要输入来自于forward()的参数V, 参数V/in_act
   * 计算结果为A/out_act, 作为下一层(l+1)的输入V/in_act 
   * 由此in_act/out_act 将前后两层数据紧密的联系在一起, 函数执行过程为:
   *    先执行: f(1) -> f(2) -> ... -> f(n)
   *    再执行: b(n) -> ... -> b(2) -> b(1)
   * 在backward()中, 梯度从最后一层(n)反向传递至第一层(1)
   * V/A的梯度仅仅在层与层之间进行传递, 实际上并未参与卷积计算
   * 
   * [数据结构]
   * 关于4类Volume的数据结构, filters 是一个四维的数据结构, 由filters 贯穿了in_act和out_act
   * 其中in_sx/out_sx, in_sy/out_sy 可以看做近似一致
   * 
   * V/in_act =  [ in_sx  x in_sy  x in_depth ]
   * filters =   [ sx     x sy     x in_depth x out_depth ]
   * A/out_act = [ out_sx x out_sy x            out_depth ]
   * 
   * [循环分析]
   * forward()负责权重计算, backward()负责梯度更新
   * 实现中forward(), backward()函数中的6层loop, 虽在不同函数中但结构完全一致 可以不需要考虑其差异
   * 
   * 先对out_act/A这个结构进行循环, 再对filters内部结构进行循环
   * 而filter循环, 可以将坐标系映射到V/in_act上(ox, oy)以实现从V/in_act上取数的效果
   *    out_act遍历: d(out_depth) -> ay(out_sy) -> ax(out_sx) ->
   *    filters遍历: fy(sy) -> fx(sx) -> fd(in_depth)
   *    经过两层遍历 就完成了 out_sx x out_sy x out_depth x in_depth 上的所有数据
   * 
   * [数值计算]
   * 此处如采用变量替换计算方法将更加清晰 A<l-1> := V, A<l> := A, f<l> := f, bias<l> := biases
   * 
   * forward()计算:
   *    权重计算 A<l>.w[ax, ay, d]     := sum( f<l>.w[fx, fy, fd] * A<l-1>.w[ox, oy, fd] ) + bias<l>.w[d]
   * 
   * backward()计算:
   *    滤镜梯度 f<l>.  dw[fx, fy, fd] += A<l-1>.w[ox, ot, fd] * A<l>.dw[ax, ay, d]
   *    输入梯度 A<l-1>.dw[ox, ot, fd] += f<l>.w[fx, fy, fd]   * A<l>.dw[ax, ay, d] 
   *    偏置梯度 bias<l>.dw[d]         +=                        A<l>.dw[ax, ay, d]
   **/

  /** 本文件中包含所有层均用点积乘以输入, 但连接模式和权重可能不同(卷积层是全连接的并且共享权重)
   * 由于这些层的构造都非常相似所以放在同一个文件中 */
  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes: 
  // - FullyConn is fully connected dot products 
  // - ConvLayer does convolutions (so weight sharing spatially)
  // putting them together in one file because they are very similar

  /** ConvLayer(opt) - 卷积层 构造函数
   * 下面主要为其构造函数用于初始化该卷积层
   * opt 参数说明
   *    输入层参数 in_depth, in_sx, in_sy
   *    滤镜参数 sx, sy, filters (滤镜数)
   *    计算参数 pad - 边缘填充, stride - 步长
   *    衰减项 l1_decay_mul, l2_decay_mul
   *    偏置项 bias_pref */
  var ConvLayer = function(opt) {

    var opt = opt || {};

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;
    
    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    /** 初始化: 此处用filters, 注意filter里的数值是由Volume随机产生的 */
    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }
  ConvLayer.prototype = {

	
	/** forward(V, is_training) - 前向算法, is_training - 未使用 */
    forward: function(V, is_training) {
      // optimized code by @mdda that achieves 2x speedup over previous version

      this.in_act = V;
      /** 建立一个out_sx x out_sy x out_depth 默认值为0的Volume A (activation矩阵), 作为输出的激活项 out_act的临时存储 */
      var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
      
      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      /** 对于out_depth, out_y, out_x 方向进行迭代 */
      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
        	/** 初始化激活值=0, 后续将进行累加 */
            var a = 0.0;
            /** 对于滤镜fy, fx 进行2个方向的迭代 */
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                /** 检查如果不越界 进行深度迭代 */
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                	/** 层输出结果 A.w[ax, ay, d] += sum( f.w[fx, fy, fd] * V.w[ox, oy, fd] ) + biases.w[d] */
                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
                  }
                }
              }
            }
            a += this.biases.w[d];
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    /** backward() - 后向算法, 不需要参数, 因为参数在forward步骤已存入in_act */
    backward: function() {

      var V = this.in_act;
      /** 初始化梯度值V.dw = 0, 其维度与in_act一致  */
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      /** 对于out_depth, out_y, out_x 方向进行迭代 */
      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
        	/** 注意: out_act/A作为本层(l)输出, 就是后一层(l+1)的输入in_act/V, 
        	 *  由于后向方法是由后向前计算的, 所以此时A的梯度信息在这一步已存在 */
            var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            /** 对于滤镜fy, fx 进行2个方向的迭代 */
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                /** 检查如果不越界 进行深度迭代 */
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    /** 滤镜梯度 f.dw[fx, fy, fd] = sum(V.w[ox, ot, fd] * A.dw[ax, ay, d])
                	  * 输入梯度 V.dw[ox, ot, fd] = sum(f.w[fx, fy, fd] * A.dw[ax, ay, d]) 
                	  * V/A的梯度仅仅在层与层之间进行传递, 实际上并未参与 卷积计算 */
                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                    f.dw[ix2] += V.w[ix1]*chain_grad;
                    V.dw[ix1] += f.w[ix2]*chain_grad;
                  }
                }
              }
            }
            /** biases.dw[d] = sum(A.dw[ax, ay, d]) */
            this.biases.dw[d] += chain_grad;
          }
        }
      }
    },
    /** getParamsAndGrads() - 返回filter参数和梯度
     *  此处仅保存biases的权重和梯度, l1/l2衰减率 */
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    /** toJSON() - 序列化, 保存为json格式 */
    toJSON: function() {
      var json = {};
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth;
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.pad = this.pad;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    /** fromJSON() - 加载json格式数据, 反序列化恢复现场 */
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
      this.filters = [];
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  /** FullyConnLayer(opt) 全连接层 构造函数
   * 下面主要为其构造函数用于初始化该全连接层
   * opt 参数说明
   *    输入项 in_sx, in_sy, in_depth
   *    输出项 out_sx, out_sy, out_depth
   *    其他项 num_neurons, filters, l1_decay_mul, l2_decay_mul, bias_pref
   * 
   * [概述]
   *    经过几轮卷积/pooling计算之后, 网络的高级推理部分由全连接层完成
   *    全连接层将获取前一层的所有神经元, 将其连接到本层所有神经元上
   *    全连接层不再用空间方式组织(只能看做是一维), 因而其后的各层也不能再用卷积计算
   * 
   * [成员变量]
   *    基础变量 layer_type : 'fc'
   *    输入变量 num_inputs, in_act
   *    输出变量 out_sx(1), out_sy(1), out_depth, out_act
   *    其他变量 filters, biases, l1_decay_mul, l2_decay_mul
   *    
   * 实现BP Propagation的卷积神经网络层 
   * 
   * [数据结构]
   * 输出层的深度out_depth由神经元数量num_neurons决定
   * 
   * V/in_act =  [ in_sx  x in_sy  x in_depth ]
   * filters =   [ 1      x 1      x (in_sx x in_sy x in_depth) x out_depth ]
   * A/out_act = [ 1      x 1                                   x out_depth ]
   * 
   * [循环分析]
   * 循环采用双层遍历, 对于forward(), backward()循环结构是一致的, 但计算不同
   *    输出项遍历: out_depth(i) -> 
   *    输入项遍历: num_inputs(d) == (in_sx x in_sy x in_depth)
   * 
   * [数值计算]
   * 比之卷积层, 此处的计算非常相似, forward()负责权重计算, backward()负责梯度更新
   * 不同的是, 在卷积层中有更明确的空间概念, 而这里把所有信息都拉平成为一个维度
   * 
   * forward()计算:
   *    权重计算 A<l>.w[i]           := sum( A<l-1>.w[d] * f<l>[i].w[d] ) + bias<l>.w[i]
   * 
   * backward()计算:
   *    滤镜梯度 f<l>[i].dw[d]       += A<l-1>.w[d]  * A<l>.dw[i]
   *    输入梯度 A<l-1>.dw[d]        += f<l>[i].w[d] * A<l>.dw[i]
   *    偏置梯度 bias<l>.dw[i]       +=                A<l>.dw[i]
   */
  var FullyConnLayer = function(opt) {
    var opt = opt || {};

    // required
    // ok fine we will allow 'filters' as the word as well
    this.out_depth = typeof opt.num_neurons !== 'undefined' ? opt.num_neurons : opt.filters;

    // optional 
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
    this.out_sx = 1;
    this.out_sy = 1;
    this.layer_type = 'fc';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth ;i++) { this.filters.push(new Vol(1, 1, this.num_inputs)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }

  FullyConnLayer.prototype = {
    forward: function(V, is_training) {
      this.in_act = V;
      var A = new Vol(1, 1, this.out_depth, 0.0);
      var Vw = V.w;
      for(var i=0;i<this.out_depth;i++) {
        var a = 0.0;
        var wi = this.filters[i].w;
        for(var d=0;d<this.num_inputs;d++) {
          a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
        }
        a += this.biases.w[i];
        A.w[i] = a;
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() {
      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
      
      // compute gradient wrt weights and data
      for(var i=0;i<this.out_depth;i++) {
        var tfi = this.filters[i];
        var chain_grad = this.out_act.dw[i];
        for(var d=0;d<this.num_inputs;d++) {
          V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
          tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
        }
        this.biases.dw[i] += chain_grad;
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.num_inputs = this.num_inputs;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.num_inputs = json.num_inputs;
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      this.filters = [];
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }

  global.ConvLayer = ConvLayer;
  global.FullyConnLayer = FullyConnLayer;
  
})(convnetjs);
