package mathfoundation.automaticdifferentiation

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager

fun controlflow(n:Int,x:NDArray): NDArray {
    var y=x
    if (n>2){
        for (i in 0..n){
            y=y.mul(x)
        }
        return y
    }else{
        y=x.mul(x)
        return y
    }
}
fun main(){
    var manager= NDManager.newBaseManager()
    var x=manager.arange(4f)
    x.setRequiresGradient(true)

    var gc=Engine.getInstance().newGradientCollector()
    var y=controlflow(3,x)
    gc.backward(y)
    gc.close()

    println("Gradient:${x.gradient}")
}