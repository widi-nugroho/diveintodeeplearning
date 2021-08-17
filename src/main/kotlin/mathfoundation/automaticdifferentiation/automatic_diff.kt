package preliminaries.automatic_differentiation_2_5

import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager

fun main(){
    var manager = NDManager.newBaseManager();
    var x = manager.arange(4f);
    x.setRequiresGradient(true)
    println(x)
    println("gradient")
    println(x.gradient)

    // y=2xx
    var gc = Engine.getInstance().newGradientCollector()
    var y = x.dot(x).mul(2);
    println("y:"+ y.toString());
    gc.backward(y);
    gc.close()
    var grad=x.getGradient()
    println("gardient")
    println(grad)

    // check gradient=4x
    var check=x.gradient.eq(x.mul(4))
    println(check)
}