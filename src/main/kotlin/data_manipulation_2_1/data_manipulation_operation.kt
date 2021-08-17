package preliminaries.data_manipulation_2_1

import ai.djl.Device
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun main(){
    var manager = NDManager.newBaseManager();

    var x = manager.create(floatArrayOf(1f, 2f, 4f, 8f));
    var y = manager.create(floatArrayOf(2f, 2f, 2f, 2f));
    var xaddy=x.add(y)
    println(xaddy)

    var xpowy=x.pow(y)
    println(xpowy)

    var xexp=x.exp()
    println(xexp)

    x = manager.arange(12f).reshape(3, 4);
    y = manager.create(floatArrayOf(2f, 1f, 4f, 3f, 1f, 2f, 3f, 4f, 4f, 3f, 2f, 1f), Shape(3, 4));
    var xconacty_axis_0=x.concat(y) // default axis = 0
    println(xconacty_axis_0)

    var xconacty_axis_1=x.concat(y, 1) // axis = 1
    println(xconacty_axis_1)

    var xeqy=x.eq(y)
    println(xeqy)

    var xsum=x.sum()
    println(xsum)

    // broadcasting mechanism
    var a = manager.arange(3f).reshape(3, 1);
    var b = manager.arange(2f).reshape(1, 2);
    var a_add_b_broadcasting=a.add(b)
    println(a_add_b_broadcasting)
}