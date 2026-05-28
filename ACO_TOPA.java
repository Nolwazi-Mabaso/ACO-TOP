import java.util.*;
import java.io.*;

public class ACO_TOPA {
    private int numNodes;
    private int numVehicles;
    private double[][] distanceMatrix;
    private Set<Integer> globallyVisited;
    private double[] scores;
    private double Tmax;
    private double[][] pheromones;
    private double[][] heuristic;
    private double alpha = 1.2;
    private double beta = 2.0;
    private double rho = 0.3;
    private double Q = 300;
    private int depot = 0;
    private Random random;
    private double minPheromone = 0.01;  
    private double penaltyFactor = 0.5; 
    private Map<String, Integer> edgeUsageCount = new HashMap<>();  

    public ACO_TOPA(long seed, String filePath, int numVehicles, double Tmax) {
        this.random = new Random(seed);
        this.numVehicles = numVehicles;
        this.Tmax = Tmax;
        this.globallyVisited = new HashSet<>(); 
        readProblemInstance(filePath);
        initializePheromones();
        initializeHeuristic();
    }

    public void readProblemInstance(String filePath) {
        try {
            File file = new File(filePath);
            Scanner scanner = new Scanner(file);

            
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.startsWith("n")) {
                    numNodes = Integer.parseInt(line.split("\\s+")[1]);
                } else if (line.startsWith("m")) {
                    numVehicles = Integer.parseInt(line.split("\\s+")[1]);
                } else if (line.startsWith("tmax")) {
                    Tmax = Double.parseDouble(line.split("\\s+")[1]);
                    break; 
                }
            }

            
            double[][] coordinates = new double[numNodes][2];
            scores = new double[numNodes];

            
            int index = 0;
            while (scanner.hasNextLine() && index < numNodes) {
                String line = scanner.nextLine().trim();
                if (!line.isEmpty()) {
                    String[] parts = line.split("\\s+");
                    coordinates[index][0] = Double.parseDouble(parts[0]); 
                    coordinates[index][1] = Double.parseDouble(parts[1]); 
                    scores[index] = Double.parseDouble(parts[2]); 
                    index++;
                }
            }

            distanceMatrix = new double[numNodes][numNodes];
            for (int i = 0; i < numNodes; i++) {
                for (int j = 0; j < numNodes; j++) {
                    if (i == j)
                        continue;
                    double dx = coordinates[i][0] - coordinates[j][0];
                    double dy = coordinates[i][1] - coordinates[j][1];
                    distanceMatrix[i][j] = Math.sqrt(dx * dx + dy * dy);
                }
            }

            scanner.close();
        } catch (Exception e) {
            System.out.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void initializePheromones() {
        pheromones = new double[numNodes][numNodes];
        for (int i = 0; i < numNodes; i++) {
            Arrays.fill(pheromones[i], 0.1);
        }
    }

    private void initializeHeuristic() {
        heuristic = new double[numNodes][numNodes];
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                if (i != j && distanceMatrix[i][j] > 0) {
                    heuristic[i][j] = scores[j] / distanceMatrix[i][j];
                }
            }
        }
    }

    public List<Solution> runACO(int iterations) {
        List<Solution> bestSolutions = new ArrayList<>();
        for (int i = 0; i < numVehicles; i++) {
            bestSolutions.add(new Solution());
        }

        for (int iter = 0; iter < iterations; iter++) {
            globallyVisited = new HashSet<>(); 
            globallyVisited.add(depot); 

            List<Solution> antSolutions = new ArrayList<>();

            for (int k = 0; k < numVehicles; k++) {
                Solution solution = constructAntSolution(k + 1, iter + 1);
                antSolutions.add(solution);

                
                if (solution.totalScore > bestSolutions.get(k).totalScore) {
                    bestSolutions.set(k, solution);
                    localSearch(bestSolutions.get(k));
                }
            }

            updatePheromones(antSolutions);
        }
        return bestSolutions;
    }

    private Solution constructAntSolution(int antId, int iteration) {
        Solution solution = new Solution();
        solution.vehicleId = antId;
        int currentNode = depot;  
        solution.route.add(currentNode);
        Set<Integer> antVisited = new HashSet<>();
        antVisited.add(currentNode);
        double currentTime = 0;
    
        while (true) {
            
            final int current = currentNode;
            List<Integer> feasibleNodes = getFeasibleNodes(currentNode, antVisited, currentTime);
            if (feasibleNodes.isEmpty()) break;
    
            
            double explorationProbability = 0.3 * (1 - (double)iteration/100);
            if (random.nextDouble() < explorationProbability) {
                
                feasibleNodes.sort(Comparator.comparingDouble(a -> {
                    String edgeKey = Math.min(current, a) + "-" + Math.max(current, a);
                    return edgeUsageCount.getOrDefault(edgeKey, 0);
                }));
            }
    
            int nextNode = selectNextNode(currentNode, feasibleNodes);
            double predictedTime = currentTime + distanceMatrix[currentNode][nextNode]
                    + distanceMatrix[nextNode][depot];
    
            if (predictedTime > Tmax) break;
    
            synchronized (globallyVisited) {
                if (!globallyVisited.contains(nextNode)) {
                    solution.route.add(nextNode);
                    solution.totalScore += scores[nextNode];
                    currentTime += distanceMatrix[currentNode][nextNode];
                    antVisited.add(nextNode);
                    globallyVisited.add(nextNode);
                    
                    
                    String edgeKey = Math.min(currentNode, nextNode) + "-" + 
                                   Math.max(currentNode, nextNode);
                    edgeUsageCount.put(edgeKey, 
                        edgeUsageCount.getOrDefault(edgeKey, 0) + 1);
                    
                    currentNode = nextNode;  
                }
            }
        }
    
        if (currentNode != depot) {
            currentTime += distanceMatrix[currentNode][depot];
            solution.route.add(depot);
        }
    
        solution.totalTime = currentTime;
        return solution;
    }
    private List<Integer> getFeasibleNodes(int currentNode, Set<Integer> antVisited, double currentTime) {
        List<Integer> feasible = new ArrayList<>();
        for (int j = 0; j < numNodes; j++) {
            if (j != currentNode && !antVisited.contains(j)) {
                synchronized (globallyVisited) {
                    if (!globallyVisited.contains(j)) {
                        double newTime = currentTime + distanceMatrix[currentNode][j];
                        if (newTime <= Tmax) {
                            feasible.add(j);
                        }
                    }
                }
            }
        }
        return feasible;
    }

private double calculateProbability(int currentNode, int nextNode, List<Integer> feasibleNodes) {
    double numerator = Math.pow(pheromones[currentNode][nextNode], alpha) * 
                       Math.pow(1.0 / distanceMatrix[currentNode][nextNode], beta);

    double denominator = 0.0;
    for (int k : feasibleNodes) {
        denominator += Math.pow(pheromones[currentNode][k], alpha) * 
                       Math.pow(1.0 / distanceMatrix[currentNode][k], beta);
    }

    return (denominator != 0) ? (numerator / denominator) : 0;
}


private int selectNextNode(int currentNode, List<Integer> feasibleNodes) {
    double[] probabilities = new double[feasibleNodes.size()];
    double sum = 0;


    for (int i = 0; i < feasibleNodes.size(); i++) {
        int nextNode = feasibleNodes.get(i);
        probabilities[i] = calculateProbability(currentNode, nextNode, feasibleNodes);
        sum += probabilities[i];
    }


    double rand = random.nextDouble() * sum;
    double cumulative = 0;

    for (int i = 0; i < feasibleNodes.size(); i++) {
        cumulative += probabilities[i];
        if (rand <= cumulative) {
            return feasibleNodes.get(i);
        }
    }

    return feasibleNodes.get(feasibleNodes.size() - 1);
}

    
    public void localSearch(Solution solution) {
        boolean improvement = true;
        while (improvement) {
            improvement = false;
            for (int i = 1; i < solution.route.size() - 2; i++) {
                for (int j = i + 1; j < solution.route.size() - 1; j++) {
                    double oldDistance = distanceMatrix[solution.route.get(i - 1)][solution.route.get(i)] +
                            distanceMatrix[solution.route.get(j)][solution.route.get(j + 1)];
                    double newDistance = distanceMatrix[solution.route.get(i - 1)][solution.route.get(j)] +
                            distanceMatrix[solution.route.get(i)][solution.route.get(j + 1)];
                    if (newDistance < oldDistance) {
                        Collections.reverse(solution.route.subList(i, j + 1));
                        improvement = true;
                    }
                }
            }
        }
    }

    private void updatePheromones(List<Solution> solutions) {
        
        edgeUsageCount.clear();
        
        for (Solution solution : solutions) {
            for (int i = 0; i < solution.route.size() - 1; i++) {
                int from = solution.route.get(i);
                int to = solution.route.get(i + 1);
                String edgeKey = Math.min(from, to) + "-" + Math.max(from, to);
                edgeUsageCount.put(edgeKey, edgeUsageCount.getOrDefault(edgeKey, 0) + 1);
            }
        }

        
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                pheromones[i][j] *= (1 - rho);
                pheromones[i][j] = Math.max(pheromones[i][j], minPheromone);
            }
        }

        
        for (Solution solution : solutions) {
            double deposit = Q / solution.totalTime;
            
            for (int i = 0; i < solution.route.size() - 1; i++) {
                int from = solution.route.get(i);
                int to = solution.route.get(i + 1);
                String edgeKey = Math.min(from, to) + "-" + Math.max(from, to);
                
                
                if (edgeUsageCount.getOrDefault(edgeKey, 0) > numVehicles/2) {
                    double penalty = penaltyFactor * deposit;
                    pheromones[from][to] = Math.max(minPheromone, pheromones[from][to] - penalty);
                    pheromones[to][from] = Math.max(minPheromone, pheromones[to][from] - penalty);
                } else {
                    
                    pheromones[from][to] += deposit;
                    pheromones[to][from] += deposit;
                }
            }
        }
    }
    class Solution {
        int vehicleId;
        List<Integer> route = new ArrayList<>(); 
        double totalScore = 0;
        double totalTime = 0;

        public double calculateDistance() {
            double distance = 0.0;
            for (int i = 0; i < route.size() - 1; i++) {
                int from = route.get(i);
                int to = route.get(i + 1);
                distance += distanceMatrix[from][to];
            }
            return distance;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Vehicle ").append(vehicleId).append(": \n");
            sb.append("  Route: ").append(route).append(" \n");
            sb.append(String.format("  Score: %.0f | Time: %.1fh | Distance: %.0fkm",
                    totalScore, totalTime, calculateDistance()));
            return sb.toString();
        }
    }
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        String outputFile = "C:/Users/HP/Documents/Assignment/run.txt";

        while (true) {
            System.out.println("\nSelect an option:");
            System.out.println("1. Search for a Seed");
            System.out.println("2. Run ACO Algorithm");
            System.out.println("3. Exit");
            System.out.print("Enter your choice: ");

            String choice = input.nextLine();

            switch (choice) {
                case "1":
                    searchSeed(outputFile, input);
                    break;
                case "2":
                    runACOAlgorithm(input, outputFile);
                    break;
                case "3":
                    System.out.println("Exiting program. Goodbye!");
                    input.close();
                    return;
                default:
                    System.out.println("Invalid choice. Please enter 1, 2, or 3.");
            }
        }
    }

    
    private static void searchSeed(String outputFile, Scanner input) {
        System.out.print("\nEnter a seed value to search: ");
        String userInput = input.nextLine();

        try {
            long seedToFind = Long.parseLong(userInput);
            boolean found = false;

            try (BufferedReader reader = new BufferedReader(new FileReader(outputFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("Seed: " + seedToFind)) {
                        found = true;
                        System.out.println("\nRun Details for Seed: " + seedToFind);
                        while ((line = reader.readLine()) != null && !line.contains("====")) {
                            System.out.println(line);
                        }
                        break;
                    }
                }
            } catch (IOException e) {
                System.out.println("Error reading file: " + e.getMessage());
            }

            if (!found) {
                System.out.println("Seed not found. Try another seed or enter new details to append.");
                System.out.print("Enter additional details to append: ");
                String newDetails = input.nextLine();

                try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true))) {
                    writer.newLine();
                    writer.write("Additional Details for Seed " + seedToFind + ": " + newDetails);
                    writer.newLine();
                    System.out.println("Details appended successfully!");
                } catch (IOException e) {
                    System.out.println("Error writing to file: " + e.getMessage());
                }
            }

        } catch (NumberFormatException e) {
            System.out.println("Invalid input. Please enter a valid seed.");
        }
    }

    
    private static void runACOAlgorithm(Scanner input, String outputFile) {
        System.out.print("Enter full file path: ");
        String filePath = input.nextLine();

        System.out.print("Enter number of vehicles: ");
        int numVehicles = Integer.parseInt(input.nextLine());

        System.out.print("Enter Tmax: ");
        double Tmax = Double.parseDouble(input.nextLine());

        List<List<Solution>> allRuns = new ArrayList<>();
        List<Long> seeds = new ArrayList<>();
        List<Solution> bestRun = null;
        double bestTotalScore = 0;
        long bestSeed = 0;

        System.out.println("\nRunning 10 trials...\n");

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile, true))) { // Append mode
            writer.println("\nACO-TOP Algorithm Results");
            writer.println("Problem File: " + filePath);
            writer.println("Number of Vehicles: " + numVehicles);
            writer.println("Tmax: " + Tmax);
            writer.println("----------------------------------------");

            for (int run = 1; run <= 10; run++) {
                long seed = new Random().nextLong();
                seeds.add(seed);
                ACO_TOPA aco = new ACO_TOPA(seed, filePath, numVehicles, Tmax);

                long startTime = System.currentTimeMillis();
                List<Solution> solutions = aco.runACO(100);
                long runtime = System.currentTimeMillis() - startTime;

                allRuns.add(solutions);
                double runTotalScore = solutions.stream().mapToDouble(s -> s.totalScore).sum();

                System.out.println("==== Run " + run + " (Seed: " + seed + ") ====");
                for (Solution solution : solutions) {
                    System.out.println(solution);
                }
                System.out.println(String.format("Total Score: %.0f | Runtime: %dms", runTotalScore, runtime));
                System.out.println("--------------------------------------------------");

                writer.println("\n==== Run " + run + " (Seed: " + seed + ") ====");
                for (Solution solution : solutions) {
                    writer.println(solution);
                }
                writer.println(String.format("Total Score: %.0f | Runtime: %dms", runTotalScore, runtime));
                writer.println("--------------------------------------------------");

                if (runTotalScore > bestTotalScore) {
                    bestTotalScore = runTotalScore;
                    bestRun = solutions;
                    bestSeed = seed;
                }
            }

            writer.println("\nBest Overall Run (Total Score: " + bestTotalScore + "):");
            writer.println("Seed: " + bestSeed);
            for (Solution solution : bestRun) {
                writer.println(solution);
            }
            writer.println("--------------------------------------------------");

            System.out.println("\nResults saved to: " + outputFile);

        } catch (IOException e) {
            System.err.println("Error writing to output file: " + e.getMessage());
        }
    }
}

