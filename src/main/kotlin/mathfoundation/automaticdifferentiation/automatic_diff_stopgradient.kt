package mathfoundation.automaticdifferentiation

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

fun main(){
    var manager= NDManager.newBaseManager()
    var x=manager.arange(4f)
    x.setRequiresGradient(true)

    var gc = Engine.getInstance().newGradientCollector()
    var y=x.mul(x)
    var z=y.mul(x)
    gc.backward(z)
    gc.close()

    println("Gradient:${x.gradient}")

    gc=Engine.getInstance().newGradientCollector()
    y=x.mul(x)
    var u=y.stopGradient()
    z=u.mul(x)
    gc.backward(z)

    println("Gradient x stop :${x.gradient}")

}