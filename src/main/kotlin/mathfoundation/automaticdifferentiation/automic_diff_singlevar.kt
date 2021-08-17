package mathfoundation.automaticdifferentiation

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

fun main(){
    var manager=NDManager.newBaseManager()
    var x=manager.arange(4f)
    x.setRequiresGradient(true)

    var gc=Engine.getInstance().newGradientCollector()
    var y=x.pow(3).add(x.mul(2)).add(3)
    println("y:"+ y.toString())
    gc.backward(y)
    gc.close()

    var gradient=x.getGradient()
    println("Gradient:$gradient")


}