package mathfoundation.automaticdifferentiation

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

fun main(){
    var manager=NDManager.newBaseManager()
    var x=manager.arange(4f)
    var y=manager.arange(4f)
    x.setRequiresGradient(true)
    y.setRequiresGradient(true)

    var gc=Engine.getInstance().newGradientCollector()
    var z=x.pow(3).add(y.pow(2))

    println("z:"+ z.toString())
    gc.backward(z)
    gc.close()

    var xgradient=x.gradient
    var ygradient=y.getGradient()
    println("X Gradient:$xgradient")
    println("Y Gradient:$ygradient")
}