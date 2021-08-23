package cnn

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.convolutional.Conv2d
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.loss.Loss

fun padding(X:NDArray){
    var block = Conv2d.builder()
        .setKernelShape(Shape(3,3))
        .optPadding(Shape(1,1))
        .setFilters(1)
        .build()

    var config = DefaultTrainingConfig(Loss.l2Loss())
    var model = Model.newInstance("conv2D")
    model.block = block

    var trainer = model.newTrainer(config)
    trainer.initialize(X.shape)

    var yHat = trainer.forward(NDList(X)).singletonOrThrow()

    println(yHat.shape.slice(2))
}

fun stride(X:NDArray){
    val block = Conv2d.builder()
        .setKernelShape(Shape(3,5))
        .optPadding(Shape(0,1))//height dulu baru width
        .optStride(Shape(3,4))
        .setFilters(1)
        .build()

    var config = DefaultTrainingConfig(Loss.l2Loss())
    var model = Model.newInstance("conv2D")
    model.block = block

    var trainer = model.newTrainer(config)
    trainer.initialize(X.shape)

    var yHat = trainer.forward(NDList(X)).singletonOrThrow()

    println(yHat.shape.slice(2))
}

fun main(){
    val manager = NDManager.newBaseManager()
    var X = manager.randomUniform(0f,1.0f, Shape(1,1,8,8))
    padding(X)
    stride(X)
}