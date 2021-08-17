package linearregression.softmax.concise

import ai.djl.Device
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import linearregression.dataset.getDataset

class Lr_softmax_concise {
    var model = Model.newInstance("concise lr")
    var net = SequentialBlock()
    lateinit var l2loss: Loss
    lateinit var lrt: Tracker
    lateinit var sgd: Optimizer
    lateinit var trainer: Trainer
    lateinit var linearLayer: Linear

    constructor(lr: Float, batchSize: Int, numInputs: Int, numOutputs: Int) {
        net.add(Blocks.batchFlattenBlock(numInputs.toLong()))
        linearLayer = Linear.builder().optBias(true).setUnits(numOutputs.toLong()).build()
        net.add(linearLayer)
        model.block = net
        l2loss = Loss.l2Loss()
        lrt = Tracker.fixed(lr)
        sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()
        val config = DefaultTrainingConfig(l2loss)
            .optOptimizer(sgd)
            .optDevices(Device.getDevices())
            .addTrainingListeners(*TrainingListener.Defaults.logging())
        trainer = model.newTrainer(config)
        trainer.initialize(Shape(1, numInputs.toLong()))
        trainer.metrics = Metrics()
    }
}

/*fun main(){
    val manager = NDManager.newBaseManager()
    val batchSize = 256
    val randomShuffle = true
    val numInputs:Long = 784
    val numOutputs:Long = 10
    val numEpochs = 5
    val lr = 0.1f

    val k= Lr_softmax_concise(lr,batchSize,numInputs.toInt(),numOutputs.toInt())
    val trainingSet: ArrayDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: ArrayDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    EasyTrain.fit(k.trainer,numEpochs,trainingSet,validationSet)

    /*for (epoch in 1..numEpochs){
        System.out.printf("Epoch %d\n", epoch)
        for (batch in k.trainer.iterateDataset(trainingSet)){
            println(batch.size)
            EasyTrain.trainBatch(k.trainer,batch)

            k.trainer.step()
            batch.close()
        }
        k.trainer.notifyListeners { listener:TrainingListener->
            k.trainer
        }
    }*/
}*/
fun main(){
    val batchSize = 256
    val randomShuffle = true

// Get Training and Validation Datasets

// Get Training and Validation Datasets
    val trainingSet: RandomAccessDataset = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val validationSet: RandomAccessDataset = getDataset(Dataset.Usage.TEST, batchSize, false)

    val manager = NDManager.newBaseManager()

    val model = Model.newInstance("softmax-regression")

    val net = SequentialBlock()
    net.add(Blocks.batchFlattenBlock((28 * 28).toLong())) // flatten input
    net.add(Linear.builder().setUnits(10).build()) // set 10 output channels


    model.block = net

    val loss: Loss = Loss.softmaxCrossEntropyLoss()

    val lrt: Tracker = Tracker.fixed(0.1f)
    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(loss)
        .optOptimizer(sgd) // Optimizer
        .optDevices(Device.getDevices(1)) // single GPU
        .addEvaluator(Accuracy()) // Model Accuracy
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config)

    trainer.initialize(Shape(1, 28 * 28)) // Input Images are 28 x 28

    val metrics = Metrics()
    trainer.metrics = metrics

    val numEpochs = 5

    EasyTrain.fit(trainer, numEpochs, trainingSet, validationSet)
    trainer.trainingResult
}