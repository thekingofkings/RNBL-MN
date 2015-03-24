package rnbl;


import java.util.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayesMultinomial;


public class Node {
	Node parent;
	LinkedList<Node> children;
	Instances D;
	NaiveBayesMultinomial nbm;
	double CLL;
	
	public Node(Instances data) {
		parent = null;
		D = data;
		children = new LinkedList<Node>();
		nbm = new NaiveBayesMultinomial();
		// learn NB multinomial classifier
		try {
			nbm.buildClassifier(D);
		} catch (Exception e) {
			e.printStackTrace();
		}
		CLL = getCLL();
	}
	
	
	public ArrayList<Instances> splitData() {
		ArrayList<Instances> res = new ArrayList<>();
		for (int i = 0; i < D.numClasses(); i++)
			res.add(new Instances(D, 0));
		
		try {
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
	
	public int sizeD() {
		return D.numInstances();
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
	
	public boolean isLeaf() {
		if (children.size() == 0)
			return true;
		else
			return false;
	}
	
	
	/**
	 * Calculate the conditional log likelihood (CLL) on current node.
	 * 
	 * @return CLL(h_node|D_node) = |D| sum_j^|D| log { p(c_j|X) / sum_k p(c_k|X) }
	 * 		= |D| sum_j^|D| log{ p(c_j|X) }
	 */
	private double getCLL() {
		double cll = 0;
		try {
			for (int j = 0; j < D.numInstances(); j++) {
				Instance s = D.instance(j);
				double[] pi = nbm.distributionForInstance(s);
				for (int i = 0; i < D.numClasses(); i++) {
					cll += Math.log(pi[i]);
				}
			}
			cll *= D.numInstances();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cll;
	}
}
