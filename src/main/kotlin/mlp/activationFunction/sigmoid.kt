package mlp.activationFunction

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.nn.Activation
import org.math.plot.Plot2DPanel
import java.awt.Color
import javax.swing.JFrame


fun main(){
    val manager = NDManager.newBaseManager()
    val x = manager.arange(-8.0f, 8.0f, 0.1f)
    x.setRequiresGradient(true)

    val gc = Engine.getInstance().newGradientCollector()
    var y: NDArray = Activation.sigmoid(x)
    gc.backward(y)
    gc.close()
    var res = x.gradient

    val xx = mutableListOf<Double>()
    val yy = mutableListOf<Double>()
    val dY = mutableListOf<Double>()

    for (i in 0..x.size()-1){
        xx.add(x.getFloat(i).toDouble())
        yy.add(y.getFloat(i).toDouble())
        dY.add(res.getFloat(i).toDouble())
    }


    var X: DoubleArray = xx.toDoubleArray()
    var Y: DoubleArray = yy.toDoubleArray()

    /*for (i in Y){
        var t=i*(1-i)
        dY.add(t)
    }*/

    val plot = Plot2DPanel()
    plot.addLinePlot("Sigmoid", Color.BLUE,X,Y)
    plot.addLinePlot("dSigmoid", Color.RED,X,dY.toDoubleArray())
    plot.setAxisLabel(0,"x")
    val frame = JFrame("Sigmoid")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}