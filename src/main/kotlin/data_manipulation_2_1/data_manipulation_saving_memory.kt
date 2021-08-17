package preliminaries.data_manipulation_2_1

import ai.djl.Device
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

fun main(){
    var manager = NDManager.newBaseManager();

    var x = manager.arange(12f).reshape(3, 4);
    var y = manager.create(floatArrayOf(2f, 1f, 4f, 3f, 1f, 2f, 3f, 4f, 4f, 3f, 2f, 1f), Shape(3, 4));

    var original = manager.zeros(y.getShape());
    var actual = original.addi(x);

    var isequal = (original == actual)
    println(isequal)
}