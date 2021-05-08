package org.ml;

import java.io.IOException;

import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot; 
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;
import tech.tablesaw.plotly.traces.BoxTrace;

public class DataAnalysis {

	public static void main(String args[]){
		System.out.println("data Analysis");
		try {
			Table insurance_data = Table.read().csv("C:\\Users\\srivatsav\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\insurance.csv");
			System.out.println(insurance_data.shape());
			System.out.println(insurance_data.first(7));
			System.out.println(insurance_data.last(7));
			System.out.println(insurance_data.structure());
			System.out.println(insurance_data.summary());
			
			Layout layout1 = Layout.builder().title("Distribution of AGE").build();
			HistogramTrace trace1 = HistogramTrace.builder(insurance_data.nCol("age")).build();
			Plot.show(new Figure(layout1, trace1));
			
			Layout layout2 = Layout.builder().title("Distribution of BMI").build();
			HistogramTrace trace2 = HistogramTrace.builder(insurance_data.nCol("bmi")).build();
			Plot.show(new Figure(layout2, trace2));
			
			Layout layout3 = Layout.builder().title("Distribution of BMI").build();
			HistogramTrace trace3 = HistogramTrace.builder(insurance_data.nCol("bmi")).build();
			Plot.show(new Figure(layout3, trace3));
			
			
			Layout layout4 = Layout.builder().title("Distribution of children ").build();
			HistogramTrace trace4 = HistogramTrace.builder(insurance_data.nCol("children")).build();
			Plot.show(new Figure(layout4, trace4));
			
			Layout layout5 = Layout.builder().title("Distribution of smoker").build();
			HistogramTrace trace5 = HistogramTrace.builder(insurance_data.nCol("smoker")).build();
			Plot.show(new Figure(layout5, trace5));
			
			Layout layout6 = Layout.builder().title("Distribution of region").build();
			HistogramTrace trace6 = HistogramTrace.builder(insurance_data.nCol("region")).build();
			Plot.show(new Figure(layout6, trace6));
			
			Layout layout7 = Layout.builder().title("Distribution of charges").build();
			HistogramTrace trace7 = HistogramTrace.builder(insurance_data.nCol("charges")).build();
			Plot.show(new Figure(layout7, trace7));
			
			Layout layout8= Layout.builder().title("charges based on smoking activity").build();
			BoxTrace trace8 = BoxTrace.builder(insurance_data.categoricalColumn("smoker"),insurance_data.nCol("charges")).build();
			Plot.show(new Figure(layout8, trace8));
			
			Layout layout9= Layout.builder().title("charges based on sex").build();
			BoxTrace trace9 = BoxTrace.builder(insurance_data.categoricalColumn("sex"),insurance_data.nCol("charges")).build();
			Plot.show(new Figure(layout9, trace9));
					
			} catch (IOException e)
		{
				// TODO Auto-generated catch blocke.printStackTrace();
				e.printStackTrace();
				}
	}

	}