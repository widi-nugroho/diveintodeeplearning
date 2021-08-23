package cnn

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.pooling.Pool
import ai.djl.training.ParameterStore

fun pool2D(X:NDArray,poolShape: Shape,mode:String): NDArray? {
    val manager = NDManager.newBaseManager()
    val poolHeight = poolShape.get(0)
    val poolWidth = poolShape.get(1)

    var Y = manager.zeros(Shape(X.shape.get(0)-poolHeight+1,
                                        X.shape.get(1)-poolWidth+1))

    for (i in 0..Y.shape.get(0)-1){
        for (j in 0..Y.shape.get(1)-1){
            if (mode=="max"){
                Y.set(NDIndex(i,j),X.get("$i:${i+poolHeight},$j:${j+poolWidth}").max())
            }else if (mode=="avg"){
                Y.set(NDIndex(i,j),X.get("$i:${i+poolHeight},$j:${j+poolWidth}").mean())
            }
        }
    }
    return Y
}

fun main(){
    val manager = NDManager.newBaseManager()
    val X: NDArray = manager.arange(9f).reshape(3, 3)
    println(X)
    println(pool2D(X, Shape(2, 2), "max"))
    println(pool2D(X, Shape(2, 2), "avg"))

    //padding and stride
    var X2 = manager.arange(16f).reshape(1, 1, 4, 4);
    var block = Pool.maxPool2dBlock(Shape(3,3),Shape(3,3))//pooling window 3,3 stride 3,3
    block.initialize(manager,DataType.FLOAT32, Shape(1,1,4,4))

    var parameterStore = ParameterStore(manager,false)
    println(block.forward(parameterStore, NDList(X2),true).singletonOrThrow())

    block = Pool.maxPool2dBlock(Shape(3,3),Shape(2,2),Shape(1,1))//pooling window,stride,padding
    println(block.forward(parameterStore, NDList(X2),true).singletonOrThrow())

    X2 = X2.concat(X2.add(1),1)
    println(X2)
    block = Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2), Shape(1, 1))
    println(block.forward(parameterStore, NDList(X2), true).singletonOrThrow())
}
