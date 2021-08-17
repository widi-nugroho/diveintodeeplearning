package linearregression.scratch

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset
import linearregression.dataset.*
import org.math.plot.Plot2DPanel
import javax.swing.JFrame

class Lr_scratch(){
    lateinit var w:NDArray
    lateinit var b:NDArray
    constructor(manager:NDManager) : this() {
        w=manager.randomNormal(Shape(1))
        b=manager.randomNormal(Shape(1))
        w.setRequiresGradient(true)
        b.setRequiresGradient(true)
    }
    fun forward(x:NDArray):NDArray{
        return w.dot(x).add(b)
    }
    fun squaredloss(yp:NDArray,yt:NDArray):NDArray{
        var diff=yp.sub(yt)
        return diff.mul(diff)
    }
    fun sgd(lr:Float,batchsize:Int){
        var dw=w.gradient.mul(lr).div(batchsize)
        w.subi(dw)
        var db=b.gradient.mul(lr).div(batchsize)
        b.subi(db)
    }
    fun train(x:NDArray,yt:NDArray,lr: Float,batchsize: Int){
        var gc=Engine.getInstance().newGradientCollector()
        var yp=forward(x)
        var lossvalue=squaredloss(yp,yt)
        gc.backward(lossvalue)
        gc.close()
        sgd(lr,batchsize)
    }
}
fun computeerror(objek:Lr_scratch,xt:NDArray,yt:NDArray): Double {
    var ypredall=objek.forward(xt)
    var difftotal=objek.squaredloss(ypredall,yt)
    var res=difftotal.mean().getFloat()
    return res.toDouble()
}
fun main(){
    val manager = NDManager.newBaseManager()

    val trueW = manager.create(1f)
    val trueB = 4.2f
    val lr = 0.01f
    val batchSize=10
    val numEpochs=5
    val k=Lr_scratch(manager)
    val dp = syntheticData(manager, trueW, trueB, 1000)
    val features = dp.x
    val labels = dp.y
    val yy= mutableListOf<Double>()
    val xx= mutableListOf<Double>()

    val dataset = ArrayDataset.Builder()
        .setData(features) // Set the Features
        .optLabels(labels) // Set the Labels
        .setSampling(batchSize, false) // set the batch size and random sampling to false
        .build()
    var errorbefore=computeerror(k,features,labels)
    yy.add(errorbefore)
    for (epoch in 0 until numEpochs) {
        // Assuming the number of examples can be divided by the batch size, all
        // the examples in the training dataset are used once in one epoch
        // iteration. The features and tags of minibatch examples are given by X
        // and y respectively.
        for (batch in dataset.getData(manager)) {
            val x = batch.data.head()
            val y = batch.labels.head()
            k.train(x, y, lr, batchSize)
            batch.close()
        }
        var errorafter=computeerror(k,features,labels)
        yy.add(errorafter)
        xx.add(epoch.toDouble())
    }
    val plot = Plot2DPanel()
    plot.addLinePlot("Plot",xx.toDoubleArray(),yy.toDoubleArray())
    plot.setAxisLabel(0,"Error")
    plot.setAxisLabel(1,"Epoch")
    val frame = JFrame("Error")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true

    println("W True:$trueW")
    println("W Predict:${k.w}")
    println("B True:$trueB")
    println("B Predict:${k.b}")

}
