package dlc.parametermanagement

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.initializer.ConstantInitializer
import ai.djl.training.initializer.Initializer
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.initializer.XavierInitializer
import ai.djl.translate.NoopTranslator

fun getNet():SequentialBlock{
    var net = SequentialBlock()
    net.add(Linear.builder().setUnits(8).build())
    net.add(Activation.reluBlock())
    net.add(Linear.builder().setUnits(1).build())
    return net
}
fun getNet2(x:NDArray): SequentialBlock {
    val manager = NDManager.newBaseManager()
    val net = SequentialBlock()

    val linear1 = Linear.builder().setUnits(8).build()
    net.add(linear1)

    net.add(Activation.reluBlock())

    val linear2 = Linear.builder().setUnits(1).build()
    net.add(linear2)

    linear1.setInitializer(XavierInitializer(),Parameter.Type.WEIGHT)
    linear1.initialize(manager,DataType.FLOAT32,x.shape
    )
    linear2.setInitializer(Initializer.ZEROS,Parameter.Type.WEIGHT)
    linear2.initialize(manager,DataType.FLOAT32,x.shape)

    println(linear1.parameters.get(0).value.array)
    println(linear2.parameters.get(0).value.array)

    return net
}
fun main(){
    val manager = NDManager.newBaseManager()
    var x = manager.randomUniform(0f,1f, Shape(2,4))
    var model = Model.newInstance("lin-reg")

    var net = SequentialBlock()
    net.add(Linear.builder().setUnits(8).build())
    net.add(Activation.reluBlock())
    net.add(Linear.builder().setUnits(1).build())
    net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
    net.initialize(manager, DataType.FLOAT32, x.shape)

    model.block = net

    net.setInitializer(ConstantInitializer(1f),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,x.shape)

    var linearlayer = net.children.get(0).value
    var weight = linearlayer.parameters.get(0).value.array
    println(weight)

    var net2 = getNet()
    net2.setInitializer(ConstantInitializer(777f),Parameter.Type.WEIGHT)
    net2.initialize(manager,DataType.FLOAT32,x.shape)
    val linearlayer2 = net2.children.get(0).value
    val weight2 = linearlayer2.parameters.get(0).value.array
    println(weight2)

    val net3 = getNet2(x)

}