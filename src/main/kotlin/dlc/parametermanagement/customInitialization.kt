package dlc.parametermanagement

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.initializer.Initializer

class customInitialization :Initializer {
    override fun initialize(manager: NDManager?, shape: Shape?, dataType: DataType?): NDArray {
        println("Init %s/n"+shape.toString())

        var data = manager!!.randomUniform(-10f,10f,shape,dataType)

        var absGte5 = data.abs().gte(5)

        return data.mul(absGte5)
    }
}

fun main(){
    val manager = NDManager.newBaseManager()
    var x = manager.randomUniform(0f,1f, Shape(2,4))

    var net = getNet()
    net.setInitializer(customInitialization(),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,x.shape)

    val linearlayer = net.children.get(0).value
    val weight = linearlayer.parameters.get(0).value.array
    println(weight)
}