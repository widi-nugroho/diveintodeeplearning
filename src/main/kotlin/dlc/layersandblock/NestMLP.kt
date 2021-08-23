package dlc.layersandblock

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.nn.core.Linear
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList

class NestMLP:AbstractBlock(2) {
    lateinit var net:SequentialBlock
    lateinit var dense:Block

    lateinit var test:Block
    init {
        net= SequentialBlock()
        net.add(Linear.builder().setUnits(64).build())
        net.add(Activation.reluBlock())
        net.add(Linear.builder().setUnits(32).build())
        net.add(Activation.reluBlock())
        addChildBlock("net",net)

        dense = addChildBlock("dense",Linear.builder().setUnits(16).build())
    }

    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList?,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var current = inputs
        current = net.forward(parameterStore,current,training)
        current = dense.forward(parameterStore,current,training)
        current = NDList(Activation.relu(current.singletonOrThrow()))
        return current
    }

    override fun getOutputShapes(inputShapes: Array<out Shape>?): Array<Shape> {
        var current = inputShapes
        for (block in children.values()){
            current = block.getOutputShapes(current)
        }
        return current as Array<Shape>
    }


    override fun initializeChildBlocks(manager: NDManager?, dataType: DataType?, vararg inputShapes: Shape?) {
        var shapes = inputShapes
        for (child in children.values()){
            child.initialize(manager,dataType,*shapes)
            shapes=child.getOutputShapes(shapes)
        }
    }
}

fun main(){
    val manager = NDManager.newBaseManager()
    val inputSize = 20
    val x = manager.randomUniform(0f, 1f, Shape(2, inputSize.toLong()))

    var chimera = SequentialBlock()

    chimera.add(NestMLP())
    chimera.add(Linear.builder().setUnits(20).build())
    chimera.add(FixedHiddenMLP(2))

    chimera.setInitializer(NormalInitializer(),Parameter.Type.WEIGHT)
    chimera.initialize(manager,DataType.FLOAT32,x.shape)

    var model = Model.newInstance("chimera")
    model.block=chimera

    val translator = NoopTranslator(null)
    val xList = NDList(x)

    var predictor = model.newPredictor(translator)
    var output = predictor.predict(xList).singletonOrThrow()
    println(output)
}