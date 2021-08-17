package preliminaries.data_manipulation_2_1

import ai.djl.Device
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun main(){
    var manager = NDManager.newBaseManager();

    var x = manager.arange(12f).reshape(3, 4);

    println("x")
    println(x)

    println("x.get(\":-1\")")
    println(x.get(":-1"))

    println("x.get(\"1:3\")")
    println(x.get("1:3"))

    println("x.get(\"1:3, 0:2\")")
    println(x.get("1:3, 0:2"))

    println("x.get(\"1:3, 1:\")")
    println(x.get("1:3, 1:"))

    // set index
    x.set(NDIndex("1, 2"), 9)
    println(x)

    x.set(NDIndex("0:2, :"), 12)
    println(x)
}