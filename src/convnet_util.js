(function(global) {
  "use strict";

  // Random number utilities
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  
  
  /** randf(a, b) - 产生一个范围在 [a, b] 之间的随机数 */
  var randf = function(a, b) { return Math.random()*(b-a)+a; }
  /** randi(a, b) - 产生一个范围在 [a, b] 之间的随机整数 */
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  /** randn(mu, std) - 产生一个均值为mu, 方差为std的随机数 */
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  /** zeros(n) - 产生一个长度为n的向量 */
  // Array utilities
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }
  
  /** arrContains(arr, elt) - 判断向量arr中是否包含elt */
  var arrContains = function(arr, elt) {
    for(var i=0,n=arr.length;i<n;i++) {
      if(arr[i]===elt) return true;
    }
    return false;
  }

  /** arrUnique(arr) - 按顺序返回arr中唯一元素组成的向量, 类似于distinct arr操作 */
  var arrUnique = function(arr) {
    var b = [];
    for(var i=0,n=arr.length;i<n;i++) {
      if(!arrContains(b, arr[i])) {
        b.push(arr[i]);
      }
    }
    return b;
  }

  /** maxmin(w) 找出一个权重向量中最大值maxv、最小值minv及其下标maxi, mini，并求得最大最小值之差dv */
  // return max and min of a given non-empty array.
  var maxmin = function(w) {
    if(w.length === 0) { return {}; } // ... ;s
    var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    var n = w.length;
    for(var i=1;i<n;i++) {
      if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
      if(w[i] < minv) { minv = w[i]; mini = i; } 
    }
    return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
  }

  /** randperm(n) - 对于0,1,2,...,n-1这个向量 给出一个随机的排列
   * 例如 randperm(3) 可能等于 [0,1,2] [0,2,1] [1,0,2]等随机排列 */
  // create random permutation of numbers, in range [0...n-1]
  var randperm = function(n) {
    var i = n,
        j = 0,
        temp;
    var array = [];
    for(var q=0;q<n;q++)array[q]=q;
    while (i--) {
        j = Math.floor(Math.random() * (i+1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
  }

  /** weightedSample(lst, probs) - 从权重列表中随机抽样某一个权重lst[k]
   * lst, probs中分别放了权重及其对应的概率 */
  // sample from list lst according to probabilities in list probs
  // the two lists are of same size, and probs adds up to 1
  var weightedSample = function(lst, probs) {
    var p = randf(0, 1.0);
    var cumprob = 0.0;
    for(var k=0,n=lst.length;k<n;k++) {
      cumprob += probs[k];
      if(p < cumprob) { return lst[k]; }
    }
  }

  /** getopt(opt, field_name, default_value) - 从对象opt中取出列名field_name所对应的值, 如不存在, 则返回default_value
   * 该功能为语法糖, 实现类似java的反射功能 */
  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if(typeof field_name === 'string') {
      // case of single string
      return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
    } else {
      // assume we are given a list of string instead
      var ret = default_value;
      for(var i=0;i<field_name.length;i++) {
        var f = field_name[i];
        if (typeof opt[f] !== 'undefined') {
          ret = opt[f]; // overwrite return value
        }
      }
      return ret;
    }
  }

  /** assert(condition, message) - 判定给定条件是否成立, 如不成立则抛出异常 message */
  function assert(condition, message) {
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  global.randf = randf;
  global.randi = randi;
  global.randn = randn;
  global.zeros = zeros;
  global.maxmin = maxmin;
  global.randperm = randperm;
  global.weightedSample = weightedSample;
  global.arrUnique = arrUnique;
  global.arrContains = arrContains;
  global.getopt = getopt;
  global.assert = assert;
  
})(convnetjs);
