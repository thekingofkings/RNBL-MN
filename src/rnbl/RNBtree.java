package rnbl;


import weka.core.Instances;

import java.io.*;
import java.util.*;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class RNBtree {
	
	Node root;
	int numClass;
	int numAttribute;
	int sizeD;
	int numNode;
	double CMDL;
	
	public RNBtree(String fname) {
		Instances data = this.loadData(fname); 
		root = new Node(data);
		numClass = data.numClasses();
		numAttribute = data.numAttributes();
		sizeD = data.numInstances();
		numNode = 1;
		CMDL = 0;
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
	
	
	/**
	 * RNBL-MN employs the conditional minimum description length (CMDL) score as stopping criterion.
	 * 
	 * @return CMDL(h|D) = sum_{node \in leaves(h)}CLL(h_node|D_node) - { log|D|/2 } * size(h);
	 */
	public double getCMDL() {
		double term2 = Math.log(numNode) / 2 * sizeh();
		double CMDL = 0;
		
		LinkedList<Node> queue = new LinkedList<>();
		queue.add(root);
		
		while (!queue.isEmpty()) {
			Node n = queue.removeFirst();
			if (n.isLeaf())
				CMDL += n.CLL;
		}
		
		CMDL -= term2;
		return CMDL;
	}
	
	
	private double sizeh() {
		return numNode * (numClass + numClass * numAttribute);
	}
	
	
	
	static public void main(String[] args) {
		RNBtree r = new RNBtree("../../lab1/reuters/ship.arff");
		System.out.printf("numAttributes: %d\nnumClasses: %d\n", r.numAttribute, r.numClass);
		
		ArrayList<Instances> res = r.root.splitData();
		for (Instances is : res) {
			System.out.printf("Number of instances: %d\n", is.numInstances());
		}
	}

}
