package dlc.parametermanagement

import ai.djl.Model
import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.repository.MRL.model
import ai.djl.training.initializer.NormalInitializer
import ai.djl.translate.NoopTranslator


fun main(){
    val manager = NDManager.newBaseManager()

    val net = SequentialBlock()
    var shared = Linear.builder().setUnits(8).build()
    var sharedRelu = SequentialBlock()
    sharedRelu.add(shared)
    sharedRelu.add(Activation.reluBlock())

    net.add(Linear.builder().setUnits(8).build())
    net.add(Activation.reluBlock())
    net.add(sharedRelu)
    net.add(sharedRelu)
    net.add(Linear.builder().setUnits(10).build())

    val x = manager.randomUniform(-10f,10f, Shape(2,20),DataType.FLOAT32)
    val model = Model.newInstance("Tied Parameter")
    net.setInitializer(NormalInitializer(),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,x.shape)

    model.setBlock(net)

    val predictor: Predictor<NDList, NDList> = model.newPredictor(NoopTranslator())
    println(predictor.predict(NDList(x)).singletonOrThrow())

// Check that the parameters are the same

// Check that the parameters are the same
    val shared1 = net.children.valueAt(2)
        .parameters.valueAt(0).array
    val shared2 = net.children.valueAt(3)
        .parameters.valueAt(0).array
    shared1.eq(shared2)
}