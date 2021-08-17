package mlp.scratch

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Batch
import ai.djl.training.dataset.Dataset
import ai.djl.training.loss.Loss
import linearregression.dataset.getDataset
import org.math.plot.Plot2DPanel
import java.awt.Color
import javax.swing.JFrame

class Mlp_scratch() {
    lateinit var w1: NDArray
    lateinit var w2: NDArray
    lateinit var b1: NDArray
    lateinit var b2: NDArray
    var numInputs: Long=0
    var numHidden: Long=0
    var numOutputs: Long=0
    lateinit var params:NDList
    var loss = Loss.softmaxCrossEntropyLoss()
    constructor(manager: NDManager,numInputs:Long,numHidden:Long,numOutputs:Long) : this() {
        this.numInputs=numInputs
        this.numHidden=numHidden
        this.numOutputs=numOutputs

        w1=manager.randomNormal(Shape(numInputs,numHidden))
        w2=manager.randomNormal(Shape(numHidden,numOutputs))
        b1=manager.randomNormal(Shape(numHidden))
        b2=manager.randomNormal(Shape(numOutputs))

        w1.setRequiresGradient(true)
        w2.setRequiresGradient(true)
        b1.setRequiresGradient(true)
        b2.setRequiresGradient(true)

        params= NDList(w1,b1,w2,b2)
    }
    private fun relu(X: NDArray): NDArray{
        return X.maximum(0f)
    }
    fun forward(x: NDArray): NDArray {
        var xr=x.reshape(Shape(-1,numInputs))
        var h = relu(xr.dot(w1).add(b1))
        return h.dot(w2).add(b2)
    }
    fun crossEntropy(yp: NDArray,yt: NDArray): NDArray {
        return yp[NDIndex(":, {}", yt.toType(DataType.INT32, false))].log().neg()
    }
    fun sgd(lr:Float,batchsize:Int){
        for(i in 0..params.size-1){
            var param=params.get(i)
            param.subi(param.gradient.mul(lr).div(batchsize))
        }
    }
    fun accuracy(yp: NDArray, yt: NDArray): Float {Int
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
    fun evaluateAccuracy(dataIterator:Iterable<Batch>): Float {
        var akurasi=0.0F
        for (batch in dataIterator){
            val X=batch.data.head().reshape(Shape(-1,numInputs))
            val Y=batch.labels.head()
            val yp=forward(X)
            val score_accuracy=accuracy(yp,Y)
            akurasi+=score_accuracy
        }
        return akurasi
    }

    fun train(x: NDArray, yt: NDArray, lr: Float, batchsize: Int){
        var gc= Engine.getInstance().newGradientCollector()
        var yp=forward(x)

        var lossvalue=crossEntropy(yp,yt)
        var l=lossvalue.mul(batchsize)
        gc.backward(l)
        gc.close()
        sgd(lr,batchsize)
    }
}

fun main(){
    val manager=NDManager.newBaseManager()
    val batchSize = 256
    val randomShuffle = true
    val numInputs:Long = 784
    val numHidden:Long = 256
    val numOutputs:Long = 10

    val xx= mutableListOf<Double>()
    val yytrain= mutableListOf<Double>()
    val yyvalidation= mutableListOf<Double>()

    val k = Mlp_scratch(manager,numInputs,numHidden,numOutputs)
    val trainingSet: ArrayDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: ArrayDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    val numEpochs = 5
    val lr = 0.1f

    for (epoch in 0 until numEpochs) {
        // Assuming the number of examples can be divided by the batch size, all
        // the examples in the training dataset are used once in one epoch
        // iteration. The features and tags of minibatch examples are given by X
        // and y respectively.
        for (batch in trainingSet.getData(manager)) {
            val x = batch.data.head()
            val y = batch.labels.head()
            k.train(x, y, lr, batchSize)
            batch.close()
        }
        var errortraining = k.evaluateAccuracy(trainingSet.getData(manager)).toDouble()
        var errorvalidation = k.evaluateAccuracy(validationSet.getData(manager)).toDouble()
        yytrain.add(errortraining/trainingSet.size())
        yyvalidation.add(errorvalidation/validationSet.size())
        xx.add(epoch.toDouble())
        println("Epoch:$epoch")
        println(errortraining/trainingSet.size())
        println(errorvalidation/validationSet.size())
    }

    val plot = Plot2DPanel()
    plot.addLinePlot("Plot Train", Color.BLUE,xx.toDoubleArray(),yytrain.toDoubleArray())
    plot.addLinePlot("Plot Validation", Color.RED,xx.toDoubleArray(),yyvalidation.toDoubleArray())
    plot.setAxisLabel(0,"Epoch")
    plot.setAxisLabel(1,"Accuracy")
    plot.addLegend("NORTH")
    val frame = JFrame("Error")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}