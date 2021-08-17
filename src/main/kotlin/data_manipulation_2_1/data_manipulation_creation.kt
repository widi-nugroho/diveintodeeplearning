package preliminaries.data_manipulation_2_1

import ai.djl.Device
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun main(){
    var manager = NDManager.newBaseManager();
    var x = manager.arange(12);
    println(x)
    x=x.reshape(3,4)
    println(x)
    x=x.reshape(2,3,2)
    println(x)
    x=manager.create(Shape(3, 4))
    println(x)

    var zeroes=manager.ones(Shape(2, 3, 4))
    println(zeroes)

    var ones=manager.ones(Shape(2, 3, 4))
    println(ones)

    var rnnormal=manager.randomNormal(10f, 2f, Shape(3, 4), DataType.FLOAT32, Device.defaultDevice())
    println(rnnormal)

    var exactvalues=manager.create(floatArrayOf(2f, 1f, 4f, 3f, 1f, 2f, 3f, 4f, 4f, 3f, 2f, 1f), Shape(3, 4))
    println(exactvalues)
}