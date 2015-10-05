package itml.agents;

import java.util.ArrayList;
import java.util.Collection;

import itml.cards.Card;
import itml.cards.Card.CardActionType;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class SimpleSuccessfulAgent extends Agent {
	
	private int countMoves = 0;
	private int m_noThisAgent;     // Index of our agent (0 or 1).
	private int m_noOpponentAgent; // Inex of opponent's agent.
	private Classifier opponentPredictionClassifier;
	private Instances opponentInstances;

	public SimpleSuccessfulAgent( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
		super(deck, msConstruct, msPerMove, msLearn);
		
		MultilayerPerceptron classifier = new MultilayerPerceptron();
		classifier.setLearningRate(0.6);
		classifier.setMomentum(0.3);
		classifier.setTrainingTime(250);
		
		opponentPredictionClassifier = classifier;
	}

	@Override
	public void startGame(int noThisAgent, StateBattle stateBattle) {
		// Remember the indicies of the agents in the StateBattle.
		countMoves = 0;
		m_noThisAgent = noThisAgent;
		m_noOpponentAgent  = (noThisAgent == 0 ) ? 1 : 0; // can assume only 2 agents battling.
	}

	@Override
	public void endGame(StateBattle stateBattle, double[] results) {
		//To change body of implemented methods use File | Settings | File Templates.
	}

	@Override
	public Card act(StateBattle stateBattle) {
		countMoves++;
		
		StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
		StateAgent opponent = stateBattle.getAgentState(m_noOpponentAgent);
		ArrayList<Card> cards = m_deck.getCards(agent.getStaminaPoints());
		
		//no learning data
		if(opponentInstances == null) {

	        // First check to see if we are in attack range, if so attack.
	        for ( Card card : cards ) {
	            if ( (card.getType() == Card.CardActionType.ctAttack) &&
	                  card.inAttackRange( agent.getCol(), agent.getRow(),
	                		  opponent.getCol(), opponent.getRow() ) ) {
	                return card;  // attack!
	            }
	        }

	        // If we cannot attack, then try to move closer to the agent.
	        Card [] move = new Card[2];
	        move[m_noOpponentAgent] = new CardRest();  

	        Card bestCard = new CardRest();
	        int  bestDistance = calcDistanceBetweenAgents( stateBattle );

	        // ... otherwise move closer to the opponent.
	        for ( Card card : cards ) {
	            StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
	            move[m_noThisAgent] = card;
	            bs.play( move );
	            int  distance = calcDistanceBetweenAgents( bs );
	            if ( distance < bestDistance ) {
	                bestCard = card;
	                bestDistance = distance;
	            }
	        }

	        return bestCard;
		}
		else {
			//Learning strategy: predict opponents move and react statically
			
			Card opponentsCard = predictOpponent(stateBattle);
			Card myCard;
			
			//try to rest
			myCard = new CardRest();
			
			//if opponent is reachable
			if(isInAttackRange(cards, agent, opponent)) {
				//if opponent wants to attack, defend
				if(opponentsCard.getType() == CardActionType.ctAttack) {
					Card cDefend = searchCardByName(cards, "cDefend");
					if(cDefend != null) {
						myCard = cDefend;
					}
					else
					{
						//move when you can't defend
						Card move = searchFirstCardByType(cards, CardActionType.ctMove);
						if(move != null) {
							myCard = move;
						}
					}
				}
				
				//if opponent has no stamina anymore to attack, attack
				if(opponent.getStaminaPoints() < 3) {
					Card cAttack = searchFirstCardByType(cards, CardActionType.ctAttack);
					if(cAttack != null) {
						myCard = cAttack;
					}
				}
			}
			else if(countMoves > 15) {
				//go closer to the opponent
				Card [] move = new Card[2];
		        move[m_noOpponentAgent] = opponentsCard;
		        int  bestDistance = calcDistanceBetweenAgents( stateBattle );
		        
				for ( Card card : cards ) {
		            StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
		            move[m_noThisAgent] = card;
		            bs.play( move );
		            int  distance = calcDistanceBetweenAgents( bs );
		            if ( distance < bestDistance ) {
		                myCard = card;
		                bestDistance = distance;
		            }
		        }
			}
			
			return myCard;
		}
	}
	
	private Card searchCardByName(ArrayList<Card> cards, String name) {
		for(Card card : cards) {
			if(card.getName().equals(name)) {
				return card;
			}
		}
		return null;
	}
	
	private Card searchFirstCardByType(Collection<Card> cards, CardActionType type) {
		for(Card card : cards) {
			if(card.getType() == type) {
				return card;
			}
		}
		return null;
	}
	
	private boolean isInAttackRange(Collection<Card> cards, StateAgent agent, StateAgent opponent) {
		for ( Card card : cards ) {
            if ( (card.getType() == Card.CardActionType.ctAttack) &&
                  card.inAttackRange( agent.getCol(), agent.getRow(), opponent.getCol(), opponent.getRow() ) ) {
                return true;  // attack!
            }
        }
		return false;
	}

	private Card predictOpponent(StateBattle stateBattle) {
		double[] values = new double[8];
		StateAgent a = stateBattle.getAgentState(0);
		StateAgent o = stateBattle.getAgentState(1);
		values[0] = a.getCol();
		values[1] = a.getRow();
		values[2] = a.getHealthPoints();
		values[3] = a.getStaminaPoints();
		values[4] = o.getCol();
		values[5] = o.getRow();
		values[6] = o.getHealthPoints();
		values[7] = o.getStaminaPoints();
		try {
			ArrayList<Card> allCards = m_deck.getCards();
			ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());
			Instance i = new Instance(1.0, values.clone());
			i.setDataset(opponentInstances);
			int out = (int)opponentPredictionClassifier.classifyInstance(i);
			Card selected = allCards.get(out);
			if(cards.contains(selected)) {
				return selected;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
	}

	@Override
	public Classifier learn(Instances instances) {
		this.opponentInstances = instances;
		try {
			opponentPredictionClassifier.buildClassifier(instances);
			Evaluation m_Evaluation = new Evaluation(instances);
		    m_Evaluation.crossValidateModel(
		    		opponentPredictionClassifier, 
		    		instances,
		    		4,
		    		instances.getRandomNumberGenerator(1));
		    System.out.println(m_Evaluation.toSummaryString());

		} catch(Exception e) {
			System.out.println("Error training classifier: " + e.toString());
		}
		return null;  //To change body of implemented methods use File | Settings | File Templates.
	}

    private int calcDistanceBetweenAgents( StateBattle bs ) {

        StateAgent asFirst = bs.getAgentState( 0 );
        StateAgent asSecond = bs.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }

}
