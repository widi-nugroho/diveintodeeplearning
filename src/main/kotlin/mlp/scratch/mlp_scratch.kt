package mlp.mlp_4_2

import ai.djl.Device
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.Dataset
import ai.djl.training.loss.Loss
import java.util.*


fun relu(X: NDArray): NDArray {
    return X.maximum(0f)
}

fun net(X: NDArray, numInputs:Long, W1:NDArray, b1:NDArray, W2:NDArray, b2:NDArray): NDArray {
    var X = X
    X = X.reshape(Shape(-1, numInputs))
    val H = relu(X.dot(W1).add(b1))
    return H.dot(W2).add(b2)
}
fun accuracy(yp: NDArray, yt: NDArray): Float {
    // Check size of 1st dimension greater than 1
    // to see if we have multiple samples
    return if (yp.shape.size(1) > 1) {
        // Argmax gets index of maximum args for given axis 1
        // Convert yHat to same dataType as y (int32)
        // Sum up number of true entries
        yp.argMax(1).toType(DataType.INT32, false).eq(yt.toType(DataType.INT32, false))
            .sum().toType(DataType.FLOAT32, false).getFloat()
    } else yp.toType(DataType.INT32, false).eq(yt.toType(DataType.INT32, false))
        .sum().toType(DataType.FLOAT32, false).getFloat()
}
fun sgd(lr:Float,batchsize:Int,params:NDList){
    for(i in 0..params.size-1){
        var param=params.get(i)
        param.subi(param.gradient.mul(lr).div(batchsize))
    }
}
fun main(){
    // building dataset
    val batchSize = 256
    val trainIter = FashionMnist.builder()
        .optUsage(Dataset.Usage.TRAIN)
        .setSampling(batchSize, true)
        .build()

    val testIter = FashionMnist.builder()
        .optUsage(Dataset.Usage.TEST)
        .setSampling(batchSize, true)
        .build()

    trainIter.prepare()
    testIter.prepare()

    // initializing model parameters
    val numInputs:Long = 784
    val numOutputs:Long = 10
    val numHiddens:Long = 256

    val manager = NDManager.newBaseManager()

    val W1 = manager.randomNormal(0f, 0.01f, Shape(numInputs, numHiddens), DataType.FLOAT32, Device.defaultDevice())
    val b1 = manager.zeros(Shape(numHiddens))
    val W2 = manager.randomNormal(0f, 0.01f, Shape(numHiddens, numOutputs), DataType.FLOAT32, Device.defaultDevice())
    val b2 = manager.zeros(Shape(numOutputs))

    val params = NDList(W1, b1, W2, b2)

    for (param in params) {
        param.setRequiresGradient(true)
    }

    // loss function
    val loss: Loss = Loss.softmaxCrossEntropyLoss()


    // training
    val numEpochs = 10
    val lr = 0.5f

    val trainLoss: DoubleArray
    val testAccuracy: DoubleArray
    val epochCount: DoubleArray
    val trainAccuracy: DoubleArray

    trainLoss = DoubleArray(numEpochs)
    trainAccuracy = DoubleArray(numEpochs)
    testAccuracy = DoubleArray(numEpochs)
    epochCount = DoubleArray(numEpochs)

    var epochLoss = 0f
    var accuracyVal = 0f

    for (epoch in 1..numEpochs) {
        print("Running epoch $epoch...... ")
        // Iterate over dataset
        for (batch in trainIter.getData(manager)) {
            val X = batch.data.head()
            val y = batch.labels.head()
            var gc=Engine.getInstance().newGradientCollector()
            val yHat = net(X, numInputs, W1, b1, W2, b2) // net function call
            val lossValue = loss.evaluate(NDList(y), NDList(yHat))
            val l = lossValue.mul(batchSize)
            accuracyVal += accuracy(yHat, y)
            epochLoss += l.sum().getFloat()
            gc.backward(l) // gradient calculation
            gc.close()
            batch.close()
            sgd(lr, batchSize,params) // updater
        }
        trainLoss[epoch - 1] = (epochLoss / trainIter.size()).toDouble()
        trainAccuracy[epoch - 1] = (accuracyVal / trainIter.size()).toDouble()
        epochLoss = 0f
        accuracyVal = 0f
        // testing now
        for (batch in testIter.getData(manager)) {
            val X = batch.data.head()
            val y = batch.labels.head()
            val yHat = net(X, numInputs, W1, b1, W2, b2) // net function call
            accuracyVal += accuracy(yHat, y)
        }
        println(accuracyVal)
        testAccuracy[epoch - 1] = (accuracyVal / testIter.size()).toDouble()
        epochCount[epoch - 1] = epoch.toDouble()
        accuracyVal = 0f
        println("Finished epoch $epoch")
    }

    println("Finished training!")

    // display result

}