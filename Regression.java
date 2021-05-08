package org.ml;

import org.junit.internal.builders.AllDefaultPossibilitiesBuilder;
import org.netlib.util.doubleW;
import org.omg.CORBA.PUBLIC_MEMBER;

import com.mitchellbosecke.pebble.node.expression.GetAttributeExpression;

import java_cup.internal_error;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.pmml.jaxbbindings.Matrix;
import weka.gui.explorer.ClassifierErrorsPlotInstances;


public class Regression {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("C:\\Users\\srivatsav\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\Finaloutput.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		System.out.println(lr);
		Evaluation lreval = new Evaluation(dataset);
		lreval.evaluateModel(lr, dataset);

		System.out.println(lreval.predictions().get(12));
		
		
		System.out.println("The number of instances ae "+lreval.numInstances());
		
		System.out.println(lreval.toSummaryString());


		/*for(int i=0;i<=1338;i++)
		{
			System.out.println("For instance "+i+"  actual and predited values are "+lreval.predictions().get(i));
		
		}*/
		
		System.out.println(lreval);
		
	}
	}

