package dlc.layersandblock

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.Block
import ai.djl.nn.Parameter
import ai.djl.nn.core.Linear
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList


class FixedHiddenMLP(var version:Byte):AbstractBlock(version) {
    lateinit var hidden20:Block
    var constantParamWeight:NDArray? = null
    var constantParamBias:NDArray? = null
    init {
        hidden20 = addChildBlock("denseLayer",Linear.builder().setUnits(20).build())
    }

    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList?,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var current = inputs
        current = hidden20.forward(parameterStore,current,training)
        current = Linear.linear(current.singletonOrThrow(),constantParamWeight,constantParamBias)
        current = NDList(Activation.relu(current.singletonOrThrow()))
        current = hidden20.forward(parameterStore,current,training)

        while (current.head().abs().sum().getFloat()>1){
            current.head().divi(2)
        }
        return NDList(current.head().abs().sum())
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType?, vararg inputShapes: Shape?) {
        var shapes = inputShapes
        for (child in children.values()){
            child.initialize(manager,dataType,*shapes)
            shapes=child.getOutputShapes(shapes)
        }
        constantParamWeight = manager.randomUniform(-0.07f,0.07f,Shape(20,20))
        constantParamBias = manager.zeros(Shape(20))
    }

    override fun getOutputShapes(inputShapes: Array<out Shape>?): Array<Shape> {
        return arrayOf<Shape>(Shape(1))
    }
}

fun main(){
    val manager = NDManager.newBaseManager()
    val inputSize = 20
    val x = manager.randomUniform(0f, 1f, Shape(2, inputSize.toLong()))
    var net = FixedHiddenMLP(2)
    net.setInitializer(NormalInitializer(),Parameter.Type.WEIGHT)
    net.initialize(manager,DataType.FLOAT32,x.shape)

    var model = Model.newInstance("fixed-mlp")
    model.block=net

    val translator = NoopTranslator(null)
    val xList = NDList(x)

    var predictor = model.newPredictor(translator)
    var output = predictor.predict(xList).singletonOrThrow()
    println(output)
}
