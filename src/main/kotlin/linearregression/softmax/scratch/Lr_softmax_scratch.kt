package linearregression.softmax.scratch

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Batch
import ai.djl.training.dataset.Dataset
import linearregression.dataset.getDataset
import linearregression.scratch.computeerror
import org.math.plot.Plot2DPanel
import java.awt.Color
import javax.swing.JFrame

class Lr_softmax_scratch() {
    lateinit var w: NDArray
    lateinit var b: NDArray
    var numInputs: Int=0
    var numOutputs: Int=0
    constructor(manager: NDManager,numInputs:Int,numOutputs:Int) : this() {
        this.numInputs=numInputs
        this.numOutputs=numOutputs
        w=manager.randomNormal(Shape(numInputs.toLong(),numOutputs.toLong()))
        b=manager.randomNormal(Shape(numOutputs.toLong()))
        w.setRequiresGradient(true)
        b.setRequiresGradient(true)
    }
    fun forward(x: NDArray): NDArray {
        var t=x.dot(w).add(b)
        return softmax(t)
    }
    fun crossEntropy(yp: NDArray,yt: NDArray): NDArray {
        return yp[NDIndex(":, {}", yt.toType(DataType.INT32, false))].log().neg()
    }
    fun sgd(lr:Float,batchsize:Int){
        var dw=w.gradient.mul(lr).div(batchsize)
        w.subi(dw)
        var db=b.gradient.mul(lr).div(batchsize)
        b.subi(db)
    }
    fun accuracy_manual(){
        //nanti dibuat
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
    fun evaluateAccuracy(dataIterator:Iterable<Batch>): Float {
        var akurasi=0.0F
        for (batch in dataIterator){
            val X=batch.data.head().reshape(Shape(-1,numInputs.toLong()))
            val Y=batch.labels.head()
            val yp=forward(X)
            val score_accuracy=accuracy(yp,Y)
            akurasi+=score_accuracy
        }
        return akurasi
    }
    fun train(x: NDArray, yt: NDArray, lr: Float, batchsize: Int){
        var gc= Engine.getInstance().newGradientCollector()
        var yp=forward(x.reshape(Shape(-1,numInputs.toLong())))
        var lossvalue=crossEntropy(yp,yt)
        gc.backward(lossvalue)
        gc.close()
        sgd(lr,batchsize)
    }
    fun softmax(x:NDArray): NDArray {
        val xexp=x.exp()
        val partition=xexp.sum(intArrayOf(1),true)
        return xexp.div(partition)
    }
}
fun main(){
    val manager=NDManager.newBaseManager()
    val batchSize = 256
    val randomShuffle = true
    val numInputs:Long = 784
    val numOutputs:Long = 10

    val xx= mutableListOf<Double>()
    val yytrain= mutableListOf<Double>()
    val yyvalidation= mutableListOf<Double>()
// get training and validation dataset
// get training and validation dataset
    val k=Lr_softmax_scratch(manager,numInputs.toInt(),numOutputs.toInt())
    val trainingSet: ArrayDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: ArrayDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    // create model parameters

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
        var errorvalidation =k.evaluateAccuracy(validationSet.getData(manager)).toDouble()
        yytrain.add(errortraining/trainingSet.size())
        yyvalidation.add(errorvalidation/validationSet.size())
        xx.add(epoch.toDouble())
        println("Epoch:$epoch")
    }

    val plot = Plot2DPanel()
    plot.addLinePlot("Plot Train", Color.BLUE,xx.toDoubleArray(),yytrain.toDoubleArray())
    plot.addLinePlot("Plot Validation", Color.RED,xx.toDoubleArray(),yyvalidation.toDoubleArray())
    plot.setAxisLabel(0,"Epoch")
    plot.setAxisLabel(1,"Accuracy")
    val frame = JFrame("Error")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}