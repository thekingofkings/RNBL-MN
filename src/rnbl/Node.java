package rnbl;


import java.util.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayesMultinomial;


public class Node {
	Node parent;
	LinkedList<Node> children;
	Instances D;
	
	public Node(Instances data) {
		parent = null;
		D = data;
		children = new LinkedList<Node>();
	}
	
	
	public ArrayList<Instances> learnMultiNominal() {
		ArrayList<Instances> res = new ArrayList<>();
		for (int i = 0; i < D.numClasses(); i++)
			res.add(new Instances(D, 0));
		
		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
		try {
			nbm.buildClassifier(D);
			for (int i = 0; i < D.numInstances(); i++) {
				Instance s = D.instance(i);
				double[] pi = nbm.distributionForInstance(s);
				int k = this.pickClusterIndex(pi);
				res.get(k).add(s);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return res;
	}
	
	
	private int pickClusterIndex(double[] pi) {
		int idx = 0;
		double max = 0;
		for (int i = 0; i < pi.length; i++) {
			if (pi[i] > max) {
				idx = i;
				max = pi[i];
			}
		}
		return idx;
	}
}
