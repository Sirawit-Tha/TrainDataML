# create streamlit app to load iris dataset from seaborn
import streamlit as st
import seaborn as sns
df = sns.load_dataset('iris')
df
