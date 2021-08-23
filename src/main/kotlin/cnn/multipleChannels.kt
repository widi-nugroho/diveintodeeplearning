package cnn

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun corr2D(X:NDArray,K:NDArray): NDArray? {
    val manager = NDManager.newBaseManager()
    val h = K.shape.get(0)
    val w = K.shape.get(1)

    var Y = manager.zeros(Shape(X.shape.get(0)-h+1,X.shape.get(1)-w+1))

    for (i in 0..Y.shape.get(0)-1){
        for (j in 0..Y.shape.get(1)-1){
            var temp = X.get("$i:${i+h},$j:${j+w}").mul(K)
            Y.set(NDIndex(i,j),temp.sum())
        }
    }
    return Y
}

fun corr2DMultiIn(X:NDArray,K:NDArray): NDArray? {
    val manager = NDManager.newBaseManager()
    val h = K.shape.get(0)
    val w = K.shape.get(1)

    var res = manager.zeros(Shape(X.shape.get(0)-h+1,X.shape.get(1)-w+1))
    for (i in 0..X.shape.get(0)-1){
        for (j in 0..K.shape.get(0)-1){
            if (i==j){
                res = res.add(corr2D(X.get(NDIndex(i)),K.get(NDIndex(j))))
            }
        }
    }
    return res
}

fun corrMultiInOut(X:NDArray,K:NDArray): NDArray? {
    val manager = NDManager.newBaseManager()
    val cin: Long = K.getShape().get(0)
    val h: Long = K.getShape().get(2)
    val w: Long = K.getShape().get(3)

    var res = manager.create(Shape(cin,X.shape.get(1)-h+1,X.shape.get(2)-w+1))
    for (j in 0..K.shape.get(0)-1){
        res.set(NDIndex(j), corr2DMultiIn(X,K.get(NDIndex(j))))
    }
    return res
}

fun corr2dMultiInOut1x1(X:NDArray,K:NDArray): NDArray? {
    val channelIn = X.shape.get(0)
    val height = X.shape.get(1)
    val width = X.shape.get(2)

    val channelOut: Long = K.shape.get(0)
    var x = X.reshape(channelIn, height * width);
    var k = K.reshape(channelOut, channelIn);
    var Y = k.dot(x)

    return Y.reshape(channelOut,height,width)
}

fun main(){
    val manager = NDManager.newBaseManager()
    var X = manager.create(Shape(2, 3, 3), DataType.INT32)
    X[NDIndex(0)] = manager.arange(9)
    X[NDIndex(1)] = manager.arange(1, 10)
    X = X.toType(DataType.FLOAT32, true)

    var K = manager.create(Shape(2, 2, 2), DataType.INT32)
    K[NDIndex(0)] = manager.arange(4)
    K[NDIndex(1)] = manager.arange(1, 5)
    K = K.toType(DataType.FLOAT32, true)

    println(corr2DMultiIn(X,K))

    K = NDArrays.stack(NDList(K, K.add(1), K.add(2)));
    println(corrMultiInOut(X,K))

    X = manager.randomUniform(0f, 1.0f, Shape(3, 3, 3))
    K = manager.randomUniform(0f, 1.0f, Shape(2, 3, 1, 1))

    val Y1 = corr2dMultiInOut1x1(X, K)
    val Y2 = corrMultiInOut(X, K)
    println(Math.abs(Y1!!.sum().getFloat()-Y2!!.sum().getFloat())<1e-6)
}