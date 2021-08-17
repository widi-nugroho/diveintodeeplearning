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
  val y: NDArray = Activation.relu(x)
  gc.backward(y)
  gc.close()
  var res=x.gradient

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

  val plot = Plot2DPanel()
  plot.addLinePlot("Relu", Color.RED,X,Y)
  plot.addLinePlot("dRelu", Color.GREEN,X,dY.toDoubleArray())
  plot.setAxisLabel(0,"x")
  plot.addLegend("NORTH")
  val frame = JFrame("Relu")
  frame.setSize(600, 600)
  frame.contentPane = plot
  frame.isVisible = true
}