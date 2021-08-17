package linearregression.concise

import ai.djl.Device
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import linearregression.dataset.syntheticData
import linearregression.scratch.Lr_scratch
import org.math.plot.Plot2DPanel
import javax.swing.JFrame

class Lr_concise {
    var model=Model.newInstance("concise lr")
    var linearLayer=Linear.builder().optBias(true).setUnits(1).build()
    var net=SequentialBlock()
    lateinit var l2loss:Loss
    lateinit var lrt:Tracker
    lateinit var sgd:Optimizer
    lateinit var trainer: Trainer
    constructor(lr:Float,batchSize:Int){
        net.add(linearLayer)
        model.block=net
        l2loss=Loss.l2Loss()
        lrt=Tracker.fixed(lr)
        sgd=Optimizer.sgd().setLearningRateTracker(lrt).build()
        val config=DefaultTrainingConfig(l2loss)
            .optOptimizer(sgd)
            .optDevices(Device.getDevices())
            .addTrainingListeners(*TrainingListener.Defaults.logging())
        trainer=model.newTrainer(config)
        trainer.initialize(Shape(batchSize.toLong(),1))
        trainer.metrics= Metrics()
    }
}

fun main(){
    val manager = NDManager.newBaseManager()

    val trueW = manager.create(1f)
    val trueB = 4.2f
    val lr = 0.01f
    val batchSize=10
    val numEpochs=5
    val dp = syntheticData(manager, trueW, trueB, 1000)
    val features = dp.x
    val labels = dp.y
    val xx= mutableListOf<Double>()
    val yy= mutableListOf<Double>()

    val dataset = ArrayDataset.Builder()
        .setData(features) // Set the Features
        .optLabels(labels) // Set the Labels
        .setSampling(batchSize, false) // set the batch size and random sampling to false
        .build()
    var k=Lr_concise(lr,batchSize)
    for (epoch in 1..numEpochs){
        System.out.printf("Epoch %d\n", epoch)
        for (batch in k.trainer.iterateDataset(dataset)){
            // Update loss and evaulator
            EasyTrain.trainBatch(k.trainer,batch)

            // Update parameters
            k.trainer.step()
            batch.close()
        }
        // reset training and validation evaluators at end of epoch
        k.trainer.notifyListeners { listener:TrainingListener->
            k.trainer
        }
        //println(k.trainer.metrics.metricNames)
        //println(k.trainer.metrics.getMetric("train_all_L2Loss"))
        //println(k.trainer.metrics.getMetric("training-metrics"))
        var t=k.trainer.metrics.getMetric("train_progress_L2Loss")
        yy.add(t[t.size-1].value.toDouble())
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

    val params=k.linearLayer.getParameters()
    var wParam=params.valueAt(0).array
    var bParam=params.valueAt(1).array
    println("True W:${trueW},LR W:${wParam}")
    println("True B:${trueB},LR B:${bParam}")
}