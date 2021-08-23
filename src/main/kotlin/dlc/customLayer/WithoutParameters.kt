package dlc.customLayer

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList
import javax.xml.crypto.Data

class WithoutParameters:AbstractBlock(2) {
    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList?,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var current = inputs!!
        return NDList(current.head().sub(current.head().mean()))
    }

    override fun getOutputShapes(inputShapes: Array<out Shape>?): Array<Shape> {
        return inputShapes as Array<Shape>
    }
}

fun main(){
    val manager = NDManager.newBaseManager()
    val layer = WithoutParameters()

    val model = Model.newInstance("centered-layer")
    model.block=layer

    val input = manager.randomUniform(-0.07f,0.07f,Shape(4,8))
    val net = SequentialBlock()
    net.add(Linear.builder().setUnits(128).build())
    net.add(layer)
    net.setInitializer(NormalInitializer(),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,input.shape)

    var predictor = model.newPredictor(NoopTranslator())
    //var input = manager.create(floatArrayOf(1f,2f,3f,4f,5f))
    var output = predictor.predict(NDList(input)).singletonOrThrow()
    println(output.mean())
}