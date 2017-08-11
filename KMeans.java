import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
	
	private final double TWSSSELimit = 0.01;
	private final int NumberOfKMeanIterations = 40;
	private HashMap<Integer, Instance> m_Centroids;
	private HashMap<Instance, Instances> m_CentroidSets;
	private int m_K;
	
	public KMeans(int k){
		setK(k);
	}
	
	public void setK(int k){
		m_K = k;
	}
	

	public void buildClusterModel(Instances instances){
		initializeCentroids(instances);
		findKMeansCentroids(instances);
	}
	
	//pick initial values randomly
	private void initializeCentroids(Instances instances){
		ArrayList<Integer> invalidNumbers = new ArrayList<Integer>(m_K);
		Random random = new Random();
		int choosenRandom = 0;
		
		m_Centroids = new HashMap<Integer, Instance>(m_K);
		
		for(int i = 0; i < m_K; i++){
			while (invalidNumbers.contains(choosenRandom = random.nextInt(instances.size()))) {
				
			}
			m_Centroids.put(choosenRandom, instances.instance(choosenRandom));
			invalidNumbers.add(choosenRandom);
		}
	}
	
	public void findKMeansCentroids(Instances instances){
		boolean fifthK = m_K == 5;
		double previousTWSSSE = Double.MAX_VALUE;
		double curentTWSSSE = Double.MAX_VALUE;
		
		if (fifthK) {
			System.out.println("K = 5");
			System.out.println("=======");
		}
		
		for(int i = 0; i < NumberOfKMeanIterations; i++){
			intializeCentroidSets(instances);
			for (Instance instance : instances) {
				if(!m_CentroidSets.containsKey(instance)){ // check if this instance is a centroid
					Instance closestCentroid = instances.instance(findClosestCentroid(instance));
					m_CentroidSets.get(closestCentroid).add(instance); //assign it to the nearest centroid
				}
			}
			
			curentTWSSSE = calcAvgWSSSE(instances);
			
			if(fifthK){
				System.out.println("Iteration number " + i + " total error is : " + curentTWSSSE);
			}
			if(Math.abs(previousTWSSSE - curentTWSSSE) < TWSSSELimit){
				if(fifthK){
					System.out.println("Iteration stopped due to low cost difference  - under than " + TWSSSELimit);
				}
				break;
			}else{
				previousTWSSSE = curentTWSSSE;
			}

			recomputeNewCentroids();
		}
	}
	
	private void recomputeNewCentroids(){
		//Re-computes the centroid for each cluster(i.e updating m_Centroids)
		for(java.util.Map.Entry<Instance, Instances> entry :m_CentroidSets.entrySet()){
			Instance centroid = entry.getKey();
			Instances cluster = entry.getValue();
			setNewCentroidFromCluster(cluster, centroid);
		}
	}
	
	private void setNewCentroidFromCluster(Instances cluster, Instance centroid){
		double sumAlpha = 0;
		double sumRed = 0;
		double sumGreen = 0;
		double sumBlue = 0;
		int numInstances = cluster.numInstances();
		
		for (Instance instance : cluster) {
			sumAlpha += instance.value(0);
			sumRed += instance.value(1);
			sumGreen += instance.value(2);
			sumBlue += instance.value(3);
		}
		
		centroid.setValue(0, (int)(sumAlpha / numInstances));
		centroid.setValue(1, (int)(sumRed / numInstances));
		centroid.setValue(2, (int)(sumGreen / numInstances));
		centroid.setValue(3, (int)(sumBlue / numInstances));
	}
	
	private void intializeCentroidSets(Instances instances){
		m_CentroidSets = new HashMap<Instance, Instances>(m_K);
		
		for (Instance centroid : m_Centroids.values()) {
			Instances emptyInstanceSet = new Instances(instances, 0);
			m_CentroidSets.put(centroid, emptyInstanceSet);
		}
	}
	
	private double calcSquaredDistanceFromCentroid(Instance x1, Instance x2){
    	double sumOfDistances = 0;
    	int dimensionOfVector = x1.numAttributes();
    	
    	for (int i = 0; i < dimensionOfVector; i++){
    		double attributeValueOfX1 = x1.value(i);
    		double attributeValueOfX2 = x2.value(i);
			sumOfDistances += Math.pow((attributeValueOfX1 - attributeValueOfX2), 2);
		}

    	return (sumOfDistances == 0) ? 0 : Math.pow(sumOfDistances, 0.5);
	}
	
	private int findClosestCentroid(Instance instance){
		int clossestIndex = 0;
		double minDistance = Double.MAX_VALUE;
		
		for(java.util.Map.Entry<Integer, Instance> entry : m_Centroids.entrySet()){
			Instance centroid = entry.getValue();
			int index = entry.getKey();
			double distance = calcSquaredDistanceFromCentroid(instance, centroid);
			if(distance < minDistance){
				minDistance = distance;
				clossestIndex = index;
			}
		}
		
		return clossestIndex;
	}
	
	public Instances quantize(Instances instances){
		Instances quantizedInstances = new Instances(instances, 0);
		
		for (Instance instance : instances) {
			Instance closestCentroid = instances.instance(findClosestCentroid(instance));
			quantizedInstances.add(closestCentroid);
		}
		
		return quantizedInstances;
	}
	
	private double calcAvgWSSSE(Instances instances){
		double WSSSE = 0;
		
		for (Instance instance : instances) {
			Instance centroid = instances.instance(findClosestCentroid(instance));
			WSSSE += calcSquaredDistanceFromCentroid(centroid, instance);
		}
		
		return WSSSE == 0 ? 0 : WSSSE / instances.numInstances(); 
	}	
}
