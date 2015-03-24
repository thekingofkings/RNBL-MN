package rnbl;


import weka.core.Instances;

import java.io.*;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class RNBtree {
	
	Node root;
	Instances data;
	
	public RNBtree(String fname) {
		data = this.loadData(fname);
	}
	
	
	private Instances loadData(String fname) {
		Instances d = null;
		try {
			// read in data from "arff" file
			BufferedReader br = new BufferedReader(new FileReader(fname));
			Instances rawd = new Instances(br);
			br.close();
			
			// set class field
			for (int i = 0; i < rawd.numAttributes(); i++) {
				String s = rawd.attribute(i).name();
				if (s.matches(".*[Cc]lass.*")) {
					System.out.printf("class label found: %s\n", s);
					rawd.setClassIndex(i);
					break;
				}
			}
			
			// filter -- String to word vector
			StringToWordVector filter = new StringToWordVector();
			filter.setInputFormat(rawd);
			d = Filter.useFilter(rawd, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return d;
	}
	
	
	
	static public void main(String[] args) {
		RNBtree r = new RNBtree("../../lab1/reuters/ship.arff");
		System.out.printf("numAttributes: %d\nnumClasses: %d\n", r.data.numAttributes(), r.data.numClasses());
	}

}
