package itml.agents;

import java.util.ArrayList;

import itml.cards.Card;
import itml.cards.Card.CardActionType;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class SimpleSuccessfulAgent extends Agent {
	
	private int m_noThisAgent;     // Index of our agent (0 or 1).
	private int m_noOpponentAgent; // Inex of opponent's agent.
	private Classifier classifier_;
	private Instances dataset;

	public SimpleSuccessfulAgent( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
		super(deck, msConstruct, msPerMove, msLearn);
		classifier_ = new J48();
	}

	@Override
	public void startGame(int noThisAgent, StateBattle stateBattle) {
		// Remember the indicies of the agents in the StateBattle.
		m_noThisAgent = noThisAgent;
		m_noOpponentAgent  = (noThisAgent == 0 ) ? 1 : 0; // can assume only 2 agents battling.
	}

	@Override
	public void endGame(StateBattle stateBattle, double[] results) {
		//To change body of implemented methods use File | Settings | File Templates.
	}

	@Override
	public Card act(StateBattle stateBattle) {
		StateAgent a = stateBattle.getAgentState(m_noThisAgent);
		StateAgent o = stateBattle.getAgentState(m_noOpponentAgent);
		ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());
		
		Card opponentsCard = predictOpponent(stateBattle);
		Card myCard;
		
		//try to rest
		myCard = new CardRest();
		
		//if opponent wants to attack, defend
		if(opponentsCard.getType() == CardActionType.ctAttack) {
			Card cDefend = searchCardByName(cards, "cDefend");
			if(cDefend != null) {
				myCard = cDefend;
			}
		}
		
		//if opponent has no stamina anymore to attack, attack
		if(o.getStaminaPoints() < 3) {
			Card cAttack = searchFirstCardByType(cards, CardActionType.ctAttack);
			if(cAttack != null) {
				myCard = cAttack;
			}
		}
		
		return myCard;
	}
	
	private Card searchCardByName(ArrayList<Card> cards, String name) {
		for(Card card : cards) {
			if(card.getName().equals(name)) {
				return card;
			}
		}
		return null;
	}
	
	private Card searchFirstCardByType(ArrayList<Card> cards, CardActionType type) {
		for(Card card : cards) {
			if(card.getType() == type) {
				return card;
			}
		}
		return null;
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
			i.setDataset(dataset);
			int out = (int)classifier_.classifyInstance(i);
			Card selected = allCards.get(out);
			if(cards.contains(selected)) {
				return selected;
			}
		} catch (Exception e) {
			System.out.println("Error classifying new instance: " + e.toString());
		}
		return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
	}

	@Override
	public Classifier learn(Instances instances) {
		this.dataset = instances;
		try {
			classifier_.buildClassifier(instances);
		} catch(Exception e) {
			System.out.println("Error training classifier: " + e.toString());
		}
		return null;  //To change body of implemented methods use File | Settings | File Templates.
	}

}
