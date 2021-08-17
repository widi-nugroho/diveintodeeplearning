package mathfoundation.automaticdifferentiation

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

fun main(){
    var manager= NDManager.newBaseManager()
    var x=manager.arange(4f)
    x.setRequiresGradient(true)

    var gc= Engine.getInstance().newGradientCollector()
    var y=x.pow(3).add(x.mul(2)).add(3)
    gc.backward(y)
    gc.close()

    var gradientsingle=x.getGradient()
    println("Gradient:$gradientsingle")

    gc==Engine.getInstance().newGradientCollector()
    var ymul=x.pow(3).add(x.mul(2)).add(3)
    ymul=x.pow(3).add(x.mul(2)).add(3)
    gc.backward(ymul)
    gc.close()

    var gradientmul=x.gradient
    println("Gradient multi invoke:$gradientmul")

}