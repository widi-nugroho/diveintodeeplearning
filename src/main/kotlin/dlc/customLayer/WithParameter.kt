package dlc.customLayer

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.XavierInitializer
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList

class WithParameter:AbstractBlock {
    lateinit var weight:Parameter
    lateinit var bias:Parameter

    var inUnits:Int = 0
    var outUnits:Int = 0

    constructor(outUnits:Int,inUnits:Int) : super(2) {
        this.inUnits=inUnits
        this.outUnits=outUnits
        weight = addParameter(
            Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .optShape(Shape(inUnits.toLong(),outUnits.toLong()))
                .build()
        )
        bias = addParameter(
            Parameter.builder()
                .setName("bias")
                .setType(Parameter.Type.BIAS)
                .optShape(Shape(outUnits.toLong()))
                .build()
        )
    }

    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList?,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var input = inputs!!.singletonOrThrow()
        var device = input.device

        var weightArr = parameterStore!!.getValue(weight,device,false)
        val biasArr = parameterStore.getValue(bias, device, false)
        return relu(linear(input, weightArr, biasArr))
    }

    fun linear(input: NDArray, weight: NDArray?, bias: NDArray?): NDArray? {
        return input.dot(weight).add(bias)
    }
    fun relu(input: NDArray?): NDList {
        return NDList(Activation.relu(input))
    }
    override fun getOutputShapes(inputs: Array<Shape?>?): Array<Shape>? {
        return arrayOf(Shape(outUnits.toLong(), inUnits.toLong()))
    }
}

fun main(){
    val manager = NDManager.newBaseManager()

    val net = SequentialBlock()
    net.add(WithParameter(8,64))
    net.add(WithParameter(1,8))

    val input = manager.randomUniform(0f,1f, Shape(2,64))

    net.setInitializer(XavierInitializer(),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,input.shape)

    val model = Model.newInstance("my-linear-custom")
    model.block=net

    val predictor = model.newPredictor(NoopTranslator())
    val output = predictor.predict(NDList(input)).singletonOrThrow()
    println(output)
}