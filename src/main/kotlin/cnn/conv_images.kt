package cnn

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.nn.convolutional.Conv2d
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.Loss

class ConvolutionalLayer{
    private lateinit var w:NDArray
    private lateinit var b:NDArray

    constructor(shape: Shape){
        var manager = NDManager.newBaseManager()
        this.w = manager.create(shape)
        this.b = manager.randomNormal(Shape(1))
        w.setRequiresGradient(true)
    }

    fun getW(): NDArray {
        return this.w
    }

    fun getB(): NDArray {
        return this.b
    }

    fun forward(X:NDArray): NDArray {
        return corr2d(X,w).add(b)
    }
}

fun corr2d(X:NDArray,K:NDArray): NDArray {
    val manager = NDManager.newBaseManager()
    var h = K.shape.get(0)
    var w = K.shape.get(1)

    var Y = manager.zeros(Shape(X.shape.get(0)-h+1,X.shape.get(1)-w+1))
    for (i in 0..Y.shape.get(0)-1){
        for (j in 0..Y.shape.get(1)-1){
            Y.set(NDIndex(i,j),X.get("$i:${i+h},$j:${j+w}").mul(K).sum())
        }
    }
    return Y
}

fun learningKernel(X:NDArray,Y:NDArray,manager:NDManager){
    var block = Conv2d.builder()
        .setKernelShape(Shape(1,2))
        .optBias(false)
        .setFilters(1)
        .build()
    block.setInitializer(NormalInitializer(),Parameter.Type.WEIGHT)
    block.initialize(manager,DataType.FLOAT32,X.shape)

    var l2loss = Loss.l2Loss()
    var params = block.parameters
    var wParam = params.get(0).value.array
    wParam.setRequiresGradient(true)

    var lossval: NDArray? = null
    var parameterStore = ParameterStore(manager,false)

    for (i in 0..9){
        wParam.setRequiresGradient(true)
        var gc = Engine.getInstance().newGradientCollector()
        var yHat = block.forward(parameterStore, NDList(X),true).singletonOrThrow()
        var l = l2loss.evaluate(NDList(Y), NDList(yHat))
        lossval=l
        gc.backward(l)
        gc.close()

        wParam.subi(wParam.gradient.mul(0.4f))

        if ((i+1)%2==0){
            println("batch ${i+1} loss: ${lossval.sum().getFloat()}")
        }
    }
    println(wParam)
}

fun main(){
    val manager = NDManager.newBaseManager()
    /*val X = manager.create(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(3, 3))
    val K = manager.create(floatArrayOf(0f, 1f, 2f, 3f), Shape(2, 2))
    println(corr2d(X, K))*/

    //deteksi garis vertikal
    var X = manager.ones(Shape(6,8))
    var XD = manager.zeros(Shape(6,8))
    for (i in 0..XD.shape.get(0)-1){
        for (j in 0..XD.shape.get(1)-1){
            if (i == j){
                XD.set(NDIndex(i,j),1f)
            }
        }
    }
    println(XD)
    X.set(NDIndex(":,2:6"),0f)
    println(X)

    val K = manager.create(floatArrayOf(1f,-1f), Shape(1,2))
    var Y = corr2d(XD,K)
    println(Y)

    X = X.reshape(1,1,6,8)
    println(X)
    Y = Y.reshape(1,1,6,7)

    learningKernel(XD.reshape(1,1,6,8),Y,manager)
}