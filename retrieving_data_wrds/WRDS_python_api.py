#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:24:26 2018

@author: thelightofking
"""

import wrds
#Tutor:https://wrds-www.wharton.upenn.edu/pages/support/accessing-wrds-remotely/accessing-wrds-remotely-python/python-code-examples-workflow-remote/
db = wrds.Connection()
def return_comp(long_string):
    temp= long_string.split(",")
    return db.get_table('comp', 'funda', columns=temp, obs=1000)
    


#/*firm variables*/
#/*income statement*/
Firm_income=return_comp('sale,revt,cogs,xsga,dp,xrd,xad,ib,ebitda,ebit,nopi,spi,pi,txp,ni,txfed,txfo,txt,xint')
#/*CF statement and others*/
cashflow=return_comp('capx,oancf,dvt,ob,gdwlia,gdwlip,gwo')
#/*assets*/
asset=return_comp('rect,act,che,ppegt,invt,at,aco,intan,ao,ppent,gdwl,fatb,fatl')
#/*liabilities*/
liability=return_comp('lct,dlc,dltt,lt,dm,dcvt,cshrc,dcpstk,pstk,ap,lco,lo,drc,drlt,txdi')
#/*equity and other*/
equity=return_comp('ceq,scstkc,emp,csho')
#/*market*/
market=return_comp('abs(prcc_f),csho,prcc_f')
