/** 
 * [概述]
 * convnetjs 是一个非常简练的卷积神经网实现库
 * 其中包含了卷积神经网中所需要的所有必要实现
 *
 * [调用框架]
 *    1: application: √ mnist.js
 *    2: trainer:        √ convnet_trainers.js
 *    3: net:               √ convnet_net.js
 *    4: layers:               √ convnet_layers_*.js
 *    5: utilities:               √ convnet_vol.js, convnet_vol_utils.js
 * 
 * [代码阅读]
 * 由于代码量相对较大, 对于mnist这个例子只需要看以下层: 
 * 输入层input, 卷积层conv, RELU, 混合层pool, 全连接层fc, softmax
 */

/** √ include: mnist.js */
/** include: util.js */
/** include: vis.js */

/** 程序中mnist页面中所用到的 convnet.js
 * 根据github上的源代码被拆解为以下部分
 * 以下为分解后的文件(按原文件的顺序给出) */

/** √ include: convnet_init.js */
/** √ include: convnet_util.js */
/** √ include: convnet_vol.js */
/** √ include: convnet_vol_util.js */
/** √ include: convnet_layers_dotproducts.js */
/** √ include: convnet_layers_pool.js */
/** √ include: convnet_layers_input.js */
/** √ include: convnet_layers_loss.js */
/** √ include: convnet_layers_nonlinearities.js */
/** include: convnet_layers_dropout.js */
/** include: convnet_layers_normalization.js */
/** √ include: convnet_net.js */
/** √ include: convnet_trainers.js */
/** include: convnet_magicnet.js */
/** √ include: convnet_export.js */

