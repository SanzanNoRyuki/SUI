#include "search-strategies.h"
#include "memusage.h"
#include <queue>
#include <stack>
#include <utility>
#include <vector>
#include "search-interface.h"
#include <set>
#include <algorithm>
#include <list>

/* Literals for memory */
inline constexpr std::size_t operator""_KB(unsigned long long v) {
    return 1024 * v;
}
inline constexpr std::size_t operator""_MB(unsigned long long v) {
    return 1024_KB * v;
}

bool operator==(const SearchState &a, const SearchState &b) {
    return a.state_ == b.state_;
}

inline constexpr size_t mem_threshold = 50_MB;

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
    struct _open {
        _open(SearchState &&new_state,  std::shared_ptr<_open> new_parent): state(new_state), parent(std::move(new_parent)), parents_executed_action(std::nullopt) {};
        _open(SearchState &&new_state,  std::shared_ptr<_open> new_parent, SearchAction new_executed_action): state(new_state), parent(std::move(new_parent)),
                                                                                                                           parents_executed_action(new_executed_action) {};
        SearchState state;
        std::shared_ptr<_open> parent;
        std::optional<SearchAction> parents_executed_action;
    };
    std::queue<std::shared_ptr<_open>> open;
    std::set<SearchState> closed;
    open.push(std::make_shared<_open>(SearchState(init_state), nullptr));
    while (!open.empty() && (mem_limit_ - getCurrentRSS()) > mem_threshold){
        std::shared_ptr<_open> current_open = open.front(); open.pop();
        SearchState current_state = current_open->state;
        if (current_state.isFinal()){ // Final state return result
            std::shared_ptr<_open> tmp(current_open);
            std::vector<SearchAction> retVal;
            while (tmp && tmp->parents_executed_action.has_value()){
                retVal.push_back(*tmp->parents_executed_action);
                tmp = tmp->parent;
            }
            std::reverse(retVal.begin(),retVal.end());
            return retVal;
        } else if (closed.find(current_state) != closed.end()){ // State has been visited
            continue;
        }

        for (const auto& action: current_state.actions()) {
            const SearchState new_state = action.execute(current_state);
            if (new_state.isFinal()){
                std::shared_ptr<_open> tmp(current_open);
                std::list<SearchAction> temp;
                temp.emplace_back(action);
                while (tmp && tmp->parents_executed_action.has_value()){
                    temp.emplace_back(*tmp->parents_executed_action);
                    tmp = tmp->parent;
                }
                std::vector<SearchAction> retVal{std::make_move_iterator(temp.rbegin()), std::make_move_iterator(temp.rend())};
                return retVal;
            } else {
                open.push(std::make_shared<_open>(SearchState(new_state),current_open, action));
            }
        }
        closed.insert(current_state);
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
    struct _open {
        SearchState state;
        unsigned next_index;
    };
    std::stack<_open> open;
    std::set<SearchState> closed;

    // Initial memory check - multiple games
    if ((mem_limit_ - getCurrentRSS()) < mem_threshold) return {};
    
    // Initialization
    SearchState current_state = init_state;
    // If dls limit is < 0 then max depth is "unlimited" (UINT32_MAX)
    uint32_t depth_limit = depth_limit_ < 0 ? UINT32_MAX : depth_limit_;
    uint32_t current_index = 0;
    uint32_t current_depth = 0;

    // Main loop
    do {
        bool dead_end = false;

        // New state
        if (current_depth <= open.size()) {
            // Final state
            if (current_state.isFinal()) {
                std::vector<SearchAction> result{};
                result.reserve(open.size());
                while (!open.empty()) {
                    _open parent = open.top(); open.pop();
                    result.push_back(parent.state.actions()[parent.next_index]);
                }
                std::reverse(result.begin(), result.end());
                return result;
            }
            // Depth limit has been reached or current state is identical to one in the past - backtrack
            else if (depth_limit < current_depth || closed.find(current_state) != closed.end()) {
                dead_end = true;
            }
        }
        // Backtracked
        else {
            current_index++;
            current_depth--;
        }

        // Go up in the tree if dead end has been reached or all actions have been explored
        if (dead_end || current_state.actions().size() <= current_index) {
            // No result found
            if (open.empty()) {
                return {};
            }
            _open parent = open.top(); open.pop();
            current_state = std::move(parent.state);
            current_index = parent.next_index;
         }
        // Go down in the tree
         else { 
            // Save current state
            open.push({current_state, current_index});
            closed.insert(current_state);

            // Next loop initialization
            current_state = std::move(current_state.actions()[current_index].execute(current_state));
            current_index = 0;
            current_depth++;
         }
    } while (mem_limit_ - getCurrentRSS() > mem_threshold);
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    double cards_not_home = king_value * colors_list.size();    // penalty for cards not home (Optimistic estimate of actions to win)
    double non_free_cells = 0;                                  // penalty for not empty free cells
    double unsorted_stack = 0;                                  // penalty for unsorted stack
    double color_variance = 0;                                  // penalty for greater color variance at homes is disadvantage (wtf??)
    double actions_to_free_next_card = 0;                       // Optimistic estimate of actions needed to get next card home
    std::vector<Card> next_cards;
    next_cards.reserve(4);
    for (const auto &home: state.homes) {
        const auto card = home.topCard();
        if(card.has_value()) {
            if (card->color == Color::Club)     // Calculate color variance (??)
                color_variance += card->value;
            else
                color_variance -= card->value;
            cards_not_home -= card->value;      // Update cards not home
            if (card->value != king_value) {    // Next card to be found in stacks
                next_cards.emplace_back(card->color, card->value + 1);
            }
        }
    }
    for (auto const &cell: state.free_cells) {  // Calculate penalty for non empty free cells
        const auto card = cell.topCard();
        if (card.has_value()) non_free_cells++;
    }
    for (const auto &stack: state.stacks) {
        double sorted = 0;
        const std::vector stack_st = stack.storage();
        for (size_t i = 0; i < stack_st.size(); ++i) {
            if ((i+1) <= stack_st.size()-1){
                if (stack_st[i].value == stack_st[i+1].value + 1)   // calculate sorted cards in deck
                    sorted++;
            }
            for (const Card &next_card:next_cards) {    // Next card found
                if (next_card == stack_st[i]){
                    actions_to_free_next_card += stack_st.size() - i;   // How much buried it is?
                }
            }
        }
        unsorted_stack += stack_st.size()-sorted;   // Number of cards in stack - sorted cards in stack
    }
    // Double penalty for cards not home.
    return (cards_not_home * 2) + unsorted_stack + (non_free_cells) /*+ std::fabs(color_variance)*/ + actions_to_free_next_card;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
    if (init_state.isFinal()) return {};
    struct _open {
        _open(SearchState &&new_state, double new_h, std::shared_ptr<_open> new_parent): state(new_state), h(new_h), g(0), parent(std::move(new_parent)), parents_executed_action(std::nullopt) {};
        _open(SearchState &&new_state, double new_h, int new_g, std::shared_ptr<_open> new_parent, SearchAction new_executed_action): state(new_state), h(new_h), g(new_g), parent(std::move(new_parent)),
                                                                                                                parents_executed_action(new_executed_action) {};
        SearchState state;
        double h; // estimate cost of path to final state
        int g; // current cost of path to state
        std::shared_ptr<_open> parent;  // for backtracking purpose
        std::optional<SearchAction> parents_executed_action; // init_state doesn't have parent, so it's std::nullopt
        inline bool operator<(const _open& r) const {
            return h + g < r.h + r.g;
        }
        inline bool operator>(const _open& r) const {
            return h + g > r.h + r.g;
        }
    };
    struct _open_greater {  // for sorting priority_queue in ascending order (h0+g0 < h1+g1 < h2+g2 < ...)
        bool operator() (const std::shared_ptr<_open> &l, const std::shared_ptr<_open> &r){
            return *l > *r;
        }
    };
    std::priority_queue<std::shared_ptr<_open>, std::vector<std::shared_ptr<_open>>, _open_greater> open;
    std::set<SearchState> closed;
    open.push(std::make_shared<_open>(SearchState(init_state), 0., nullptr));
    while (!open.empty() && (mem_limit_ - getCurrentRSS()) > mem_threshold){
        std::shared_ptr current_open(open.top());
        open.pop();
        SearchState current_state(current_open->state);

        for (const auto &action: current_state.actions()) {
            SearchState new_state(action.execute(current_state));
            if (new_state.isFinal()){    // return result
                std::shared_ptr<_open> tmp(current_open);
                std::list<SearchAction> temp;
                temp.emplace_back(action);
                while (tmp && tmp->parents_executed_action.has_value()){
                    temp.emplace_back(*tmp->parents_executed_action);
                    tmp = tmp->parent;
                }
                std::vector<SearchAction> retVal{std::make_move_iterator(temp.rbegin()), std::make_move_iterator(temp.rend())};
                return retVal;
            } else if (closed.find(current_state) != closed.end()){
                continue;
            } else {
                open.push(std::make_shared<_open>(std::move(new_state), compute_heuristic( new_state,*heuristic_), current_open->g + 1, current_open, action));
            }
        }
        closed.insert(current_open->state);
    }
	return {};
}
