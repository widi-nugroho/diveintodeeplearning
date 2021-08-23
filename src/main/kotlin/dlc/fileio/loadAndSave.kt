package dlc.fileio

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.util.Utils
import java.io.FileInputStream
import java.io.FileOutputStream

fun main(){
    val manager = NDManager.newBaseManager()
    var x = manager.arange(4)
    //write file
    var fos = FileOutputStream("/data/project/diveintodeeplearning/src/main/resources/x-file")
    fos.write(x.encode())

    var x2:NDArray
    var fis =FileInputStream("/data/project/diveintodeeplearning/src/main/resources/x-file")
    x2 = NDArray.decode(manager, Utils.toByteArray(fis))
    println(x2)
}