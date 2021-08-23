package dlc.fileio

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.translate.NoopTranslator
import ai.djl.util.PairList
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.Files


fun createMLP():SequentialBlock{
    val mlp = SequentialBlock()
    mlp.add(Linear.builder().setUnits(256).build())
    mlp.add(Activation.reluBlock())
    mlp.add(Linear.builder().setUnits(10).build())
    return mlp
}

fun main(){
    val manager = NDManager.newBaseManager()
    val original = createMLP()

    val x = manager.randomUniform(0f,1f, Shape(2,5))
    original.initialize(manager,DataType.FLOAT32,x.shape)

    val model = Model.newInstance("mlp")
    model.block=original

    var predictor = model.newPredictor(NoopTranslator())
    var y = NDList(predictor.predict(NDList(x))).singletonOrThrow()
    println(y)

    //save file
    var mlpParamFile = File("/data/project/diveintodeeplearning/src/main/resources/mlp.param")
    var os = DataOutputStream(Files.newOutputStream(mlpParamFile.toPath()))
    original.saveParameters(os)

    //load file
    var clone = createMLP()
    clone.loadParameters(manager, DataInputStream(Files.newInputStream(mlpParamFile.toPath())))

    // Original model's parameters
    // Original model's parameters
    val originalParams: PairList<String, Parameter> = original.parameters
    // Loaded model's parameters
    // Loaded model's parameters
    val loadedParams: PairList<String, Parameter> = clone.parameters

    for (i in 0 until originalParams.size()) {
        if (originalParams.valueAt(i).getArray().equals(loadedParams.valueAt(i).getArray())) {
            System.out.printf("True ")
        } else {
            System.out.printf("False ")
        }
    }
}