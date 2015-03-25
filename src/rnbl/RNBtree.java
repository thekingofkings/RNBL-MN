package rnbl;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.*;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class RNBtree extends Classifier{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2473065824130175652L;
	
	Node root;
	Instances data;
	int numClass;
	int numAttribute;
	int sizeD;
	int numNode;
	
	public RNBtree(String fname) {
		data = this.loadData(fname); 
		root = new Node(data);
		numClass = data.numClasses();
		numAttribute = data.numAttributes();
		sizeD = data.numInstances();
		numNode = 1;
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
		double term2 = Math.log(sizeD) / 2 * sizeh();
		double cmdl = 0;
		
		LinkedList<Node> queue = new LinkedList<>();
		queue.add(root);
		
		while (!queue.isEmpty()) {
			Node n = queue.removeFirst();
			if (n.isLeaf())
				cmdl += n.CLL;
		}
		cmdl = cmdl - term2;
		return cmdl;
	}
	
	
	private double sizeh() {
		return numNode * (numClass + numClass * numAttribute);
	}
	
	
	// implement the Weka classifier interface
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		LinkedList<Node> queue = new LinkedList<>();
		queue.add(root);
		double prev_cmdl = Double.NEGATIVE_INFINITY;
		double cmdl = this.getCMDL();
		Node n = null;
		ArrayList<Instances> res = null;
		while (prev_cmdl < cmdl) {
			prev_cmdl = cmdl;
			n = queue.removeFirst();
			res = n.splitNode();
			this.numNode += res.size();
			for (Node c : n.children)
				queue.add(c);
			cmdl = this.getCMDL();
		}
		n.revokeSplit();
		numNode -= res.size();
		System.out.printf("RNBtree trained with %d nodes\n.", numNode);
	}
	
	
	@Override
	public double[] distributionForInstance(Instance instance) {
		try {
			Node n = root;
			Node prev = null;
			while (n != null) {
				prev = n;
				n = n.classifyInstanceToSubtree(instance);
			}
			return prev.distributionForInstance(instance);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new double[]{0, 0};
	}
	
	
	
	static public void main(String[] args) {
		File folder = new File("../../lab1/reuters/");
		String[] files = folder.list(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return name.matches(".*\\.arff$");
			}
		});

		try {
			for (String f : files) {
				System.out.println(f);
				RNBtree r = new RNBtree("../../lab1/reuters/" + f);
				System.out.printf("numAttributes: %d\nnumClasses: %d\n", r.numAttribute, r.numClass);
				r.buildClassifier(r.data);
				Evaluation eval = new Evaluation(r.data);
				eval.crossValidateModel(r, r.data, 10, new Random(1));
				System.out.println(eval.toClassDetailsString());
				
				Evaluation e2 = new Evaluation(r.data);
				e2.crossValidateModel(new J48(), r.data, 10, new Random(1));
				System.out.println(e2.toClassDetailsString());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	

}
