#include<bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define ll long long
#define mp make_pair
#define MAX 1000006
#define mod 1000000007
#define pb push_back
#define INF 1e18
#define pii pair<int,int>

vector<int> v[MAX];
int a[MAX];
bool vis[MAX];
vector<int> path;

void dfs1(int x,int y)
{
    vis[x]=1;
    path.pb(x);
    if(x==y)
        return;
    vector<int> temp=path;
    for(int i=0;i<v[x].size();i++)
    {
        if(!vis[v[x][i]])
        {
            dfs1(v[x][i],y);
            if(!vis[y])
                path=temp;
            else
                return;
        }
    }
}

ll dfs(int x,int y)
{
    vis[x]=1;
    ll sum=(ll)a[x];
    for(int i=0;i<v[x].size();i++)
    {
        if(!vis[v[x][i]]&&v[x][i]!=y)
            sum+=dfs(v[x][i],y);
    }
    return sum;
}

int main()
{
    //freopen ("input1.in","r",stdin);
    //freopen ("output22.txt","w",stdout);
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int n,i,x,y,q,posx,posy,v1=0,v2=0;
    ll sumx,sumy,y1=0,y2=0;
    cin>>n;
    for(i=1;i<=n;i++)
        cin>>a[i];
    for(i=0;i<n-1;i++)
    {
        cin>>x>>y;
        v[x].pb(y);
        v[y].pb(x);
    }
    cin>>q;
    while(q--)
    {
        path.clear();
        cin>>x>>y;
        memset(vis,0,n+1);
        dfs1(x,y);
        posx=(path.size()-1)/2;
        posy=path.size()-posx-1;
        if(posx==posy)
            posy++;
        memset(vis,0,n+1);
        sumx=dfs(path[posx],path[posy]);
        memset(vis,0,n+1);
        sumy=dfs(path[posy],path[posx]);
        y1+=sumx;
        y2+=sumy;
        if(sumx>=sumy)
            v1++;
        else
            v2++;
    }
    cout<<v1<<" "<<y1<<" "<<v2<<" "<<y2<<endl;
    return 0;
}
