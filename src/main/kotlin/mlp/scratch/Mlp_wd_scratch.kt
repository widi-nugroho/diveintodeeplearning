package mlp.scratch

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
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

class l2_scratch {
    lateinit var w1: NDArray
    lateinit var b1: NDArray
    lateinit var w2: NDArray
    lateinit var b2: NDArray
    lateinit var params: NDList
    var numInputs:Long=0
    var numHiddens:Long=0
    var numOutputs:Long=0
    val loss: Loss = Loss.softmaxCrossEntropyLoss()

    constructor(manager: NDManager, numInputs:Long, numHiddens:Long, numOutputs:Long){
        this.numInputs = numInputs
        this.numHiddens = numHiddens
        this.numOutputs = numOutputs

        w1 = manager.randomNormal(0f, 0.01f, Shape(numInputs, numHiddens), DataType.FLOAT32, Device.defaultDevice())
        b1 = manager.zeros(Shape(numHiddens))
        w2 = manager.randomNormal(0f, 0.01f, Shape(numHiddens, numOutputs), DataType.FLOAT32, Device.defaultDevice())
        b2 = manager.zeros(Shape(numOutputs))

        params= NDList(w1,b1,w2,b2)
        for (param in params){
            param.setRequiresGradient(true)
        }
    }
    fun relu(X: NDArray): NDArray {
        return X.maximum(0f)
    }
    fun net(X: NDArray): NDArray {
        var X = X
        X = X.reshape(Shape(-1, numInputs))
        val H = relu(X.dot(w1).add(b1))
        return H.dot(w2).add(b2)
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
    fun evaluateaccuracy(dataIterator:Iterable<Batch>, lr:Float, batchsize: Int): Float {
        var epochLoss = 0f
        var accuracyVal = 0f
        for (batch in dataIterator){
            val X = batch.data.head()
            val y = batch.labels.head()
            var gc= Engine.getInstance().newGradientCollector()

            val yp = net(X)
            val lossValue = loss.evaluate(NDList(y), NDList(yp))
            val l = lossValue.mul(batchsize)
            accuracyVal+=accuracy(yp,y)
            epochLoss+=l.sum().getFloat()
            gc.backward(l)
            gc.close()
            batch.close()
            sgd(lr,batchsize,params)
        }
        return accuracyVal
    }
    fun sgd(lr:Float,batchsize:Int,params: NDList){
        for(i in 0..params.size-1){
            var param=params.get(i)
            param.subi(param.gradient.mul(lr).div(batchsize))
        }
    }
    fun l2penalty(w:NDArray,lambda:Float): NDArray {
        var w1s=w1.pow(2).sum()
        var w2s=w2.pow(2).sum()
        var ws=w1s.add(w2s)
        return ws.mul(lambda).div(2)
    }
    fun train(x: NDArray, yt: NDArray, lr: Float, batchsize: Int,lambda: Float){
        var gc= Engine.getInstance().newGradientCollector()
        var yp=net(x)
        var lossvalue=loss.evaluate(NDList(yt), NDList(yp))
        lossvalue=lossvalue.add(l2penalty(lossvalue,lambda))
        val l = lossvalue.mul(batchsize)
        gc.backward(l)
        gc.close()
        sgd(lr,batchsize,params)
    }
}

fun main(){
    val manager=NDManager.newBaseManager()
    val numInputs:Long = 784
    val numOutputs:Long = 10
    val numHiddens:Long = 256
    val batchSize = 256
    val numEpochs = 5
    val lr = 0.5f
    val lambda = 0.1F
    val randomShuffle = true

    val xepoch = mutableListOf<Double>()
    val ytraining = mutableListOf<Double>()
    val yvalidation = mutableListOf<Double>()

    val trainingSet: ArrayDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: ArrayDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    val k=l2_scratch(manager,numInputs,numHiddens,numOutputs)

    for(epoch in 0 until numEpochs){
        print("Running epoch $epoch...... ")
        for (batch in trainingSet.getData(manager)){
            val X = batch.data.head()
            val y = batch.labels.head()
            k.train(X,y,lr,batchSize,lambda)
        }
        var errortraining = k.evaluateaccuracy(trainingSet.getData(manager),lr,batchSize).toDouble()
        var errorvalidation = k.evaluateaccuracy(validationSet.getData(manager),lr,batchSize).toDouble()

        println("Epoch:$epoch")
        println(errortraining/trainingSet.size())
        println(errorvalidation/validationSet.size())
        xepoch.add(epoch.toDouble())
        ytraining.add(errortraining/trainingSet.size())
        yvalidation.add(errorvalidation/validationSet.size())
    }
    val plot = Plot2DPanel()
    plot.addLinePlot("Plot Train", Color.BLUE,xepoch.toDoubleArray(),ytraining.toDoubleArray())
    plot.addLinePlot("Plot Validation", Color.RED,xepoch.toDoubleArray(),yvalidation.toDoubleArray())
    plot.setAxisLabel(0,"Epoch")
    plot.setAxisLabel(1,"Accuracy")
    plot.addLegend("NORTH")
    val frame = JFrame("L2decay")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}