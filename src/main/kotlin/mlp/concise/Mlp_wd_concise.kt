package mlp.concise

import ai.djl.Device
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.Dataset
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.L2Loss
import ai.djl.training.loss.Loss
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import linearregression.dataset.getDataset
import org.math.plot.Plot2DPanel
import java.awt.Color
import javax.swing.JFrame

class CrossWD(var lambda:Float, var net:SequentialBlock):SoftmaxCrossEntropyLoss(){
    override fun evaluate(label: NDList?, prediction: NDList?): NDArray {
        val w1s=net.parameters.get("02Linear_weight").array.pow(2).sum()
        val w2s=net.parameters.get("04Linear_weight").array.pow(2).sum()
        val ws=w1s.add(w2s)
        val lossvalue=super.evaluate(label, prediction)
        return lossvalue.add(ws.mul(lambda))
    }
}
class Mlp_wd_concise {
    lateinit var model: Model
    lateinit var net: SequentialBlock
    lateinit var loss: Loss
    lateinit var lrt: Tracker
    lateinit var sgd: Optimizer
    lateinit var trainer: Trainer

    constructor(lr:Float,lambda: Float,numInputs:Long,numHiddens:Long,numOutputs:Long){
        model = Model.newInstance("mlp-concise")
        net = SequentialBlock()
        net.add(Blocks.batchFlattenBlock(numInputs))
        net.add(Linear.builder().setUnits(numHiddens).build())
        net.add(Activation.reluBlock())
        net.add(Linear.builder().setUnits(numOutputs).build())

        model.block=net

        loss = CrossWD(lambda,net)

        lrt= Tracker.fixed(lr)
        sgd= Optimizer.sgd().setLearningRateTracker(lrt).build()

        val config = DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer
            .optDevices(Device.getDevices(1)) // single GPU
            .addEvaluator(Accuracy()) // Model Accuracy
            .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

        trainer=model.newTrainer(config)

        trainer.initialize(Shape(1,numInputs))
        trainer.metrics= Metrics()
    }
}

fun main(){
    val manager= NDManager.newBaseManager()
    val numInputs:Long = 784
    val numOutputs:Long = 10
    val numHiddens:Long = 200
    val batchSize = 256
    val numEpochs = 10
    val lr = 0.1f
    val lambda = 0.01f
    val randomShuffle = true

    val yAccuracy = mutableListOf<Double>()
    val yLoss = mutableListOf<Double>()
    val xEpochs = mutableListOf<Double>()

    val k = Mlp_wd_concise(lr,lambda,numInputs,numHiddens,numOutputs)

    val trainingSet: RandomAccessDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: RandomAccessDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    EasyTrain.fit(k.trainer, numEpochs, trainingSet, validationSet)

    val epochAccuracy = k.trainer.metrics.getMetric("train_epoch_Accuracy")
    val epochLoss = k.trainer.metrics.getMetric("train_epoch_SoftmaxCrossEntropyLoss")

    k.trainer.trainingResult

    for (epoch in 0..numEpochs-1){
        xEpochs.add(epoch.toDouble())
        yAccuracy.add(epochAccuracy[epoch].value.toDouble())
        yLoss.add(epochLoss[epoch].value.toDouble())
    }

    val plot = Plot2DPanel()
    plot.addLinePlot("Accuracy", Color.RED,xEpochs.toDoubleArray(),yAccuracy.toDoubleArray())
    plot.addLinePlot("Loss", Color.BLUE,xEpochs.toDoubleArray(),yLoss.toDoubleArray())
    plot.addLegend("NORTH")
    plot.setAxisLabel(0,"Epoch")
    val frame = JFrame("Softmax")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}