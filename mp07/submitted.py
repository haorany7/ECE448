'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = {}
    variables=[]
    a='antecedents'
    con='consequent'
    st='something'
    st_count=0
    for rule in nonstandard_rules:
        cur_rule=nonstandard_rules[rule]
        standardized_rules[rule]={}
        standardized_rules[rule][a]=[]
        #handle antecedents of the current rule
        for cur_an in cur_rule[a]:
            new_an = cur_an.copy()
            for index,word in enumerate(cur_an):
                if word == st:
                    st_count+=1
                    cur_variable = hex(st_count)
                    new_an[index]=cur_variable
                    variables.append(cur_variable)
            standardized_rules[rule][a].append(new_an)
        #handle consequent of the current rule
        new_con = cur_rule[con].copy()
        for index,word in enumerate(cur_rule[con]):          
            if word == st:
                new_con[index]=hex(st_count)
        standardized_rules[rule][con]=new_con
        #copy the text from the corresponding nonstandard rule to the current standadized rule
        standardized_rules[rule]['text']=nonstandard_rules[rule]['text']
            
        
    return standardized_rules, variables
def substitute(a,b,query,datum):
    for index,i in enumerate(query):
        if i==a:
            query[index]=b
    for index,i in enumerate(datum):
        if i==a:
            datum[index]=b
            
def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    if (query[3] != datum[3]):
        return None, None
    unification=[]
    subs={}
    _query=query.copy()
    _datum=datum.copy()
    for i in range(len(_query)):
        if (_query[i] in variables):
            subs[_query[i]]=_datum[i]
            substitute(_query[i],_datum[i],_query,_datum)
        elif (_datum[i] in variables):
            subs[_datum[i]]=_query[i]
            substitute(_datum[i],_query[i],_query,_datum)
    unification=_query.copy()
    return unification, subs
def compare(goal,rule,variables):
    if len(goal)!=len(rule):
        return False
    for i in range(len(goal)):
        if goal[i] not in variables and rule[i] not in variables:
            if (goal[i]!=rule[i]):
                return False
    return True
def sub_ante(ante,subs):
    _ante=ante.copy()
    for index,word in enumerate(_ante):
        if word in subs:
            _ante[index]=subs[word]
    for index,word in enumerate(_ante):
        if word in subs:
            _ante[index]=subs[word]
    return _ante
def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    applications=[]
    goalsets=[]
    for index,goal in enumerate(goals):
        cur_newgoal=goals.copy()
        cur_newgoal.remove(goal)
        if (compare(goal,rule['consequent'],variables)):
            unification,subs=unify(goal,rule['consequent'],variables)
            cur_app={}
            cur_app['consequent']=unification
            cur_app['antecedents']=[]
            for ante in rule['antecedents']:
                ante_subed=sub_ante(ante,subs)
                cur_app['antecedents'].append(ante_subed)
                cur_newgoal.append(ante_subed)
            applications.append(cur_app)
            goalsets.append(cur_newgoal)
    return applications, goalsets

import queue
def tupify(newgoal):
    newgoal_tuple=[]
    for goal in newgoal:
        newgoal_tuple.append(tuple(goal))
    return tuple(newgoal_tuple)
def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    app_table={}
    proof = []
    q = queue.Queue()
    q.put([query.copy()])
    visited = set()
    visited.add(tupify([query]))
    son_link = {}
    son_link[tupify([query])] = None
    while not q.empty():
        state = q.get()
        for rule in rules.values():
            applications, newgoals = apply(rule, state, variables)
            for newgoal, application in zip(newgoals, applications):
                application['text'] = rule['text']
                if (newgoal==[]):
                    son_link['True'] = tupify(state)
                    app_table['True'] = application
                    break
                newgoal_hash = tupify(newgoal)
                if newgoal_hash not in visited:
                    q.put(newgoal.copy())
                    visited.add(newgoal_hash)
                    son_link[newgoal_hash] = tupify(state)
                    app_table[newgoal_hash] = application

    if 'True' not in son_link:
        return None

    cur = 'True'
    while cur != tupify([query]):
        proof.append(app_table[cur])
        cur = son_link[cur]           
    return proof

          
          
          
          
          
          
          