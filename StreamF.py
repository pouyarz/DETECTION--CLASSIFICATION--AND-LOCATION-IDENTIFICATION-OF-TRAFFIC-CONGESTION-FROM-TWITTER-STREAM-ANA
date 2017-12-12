import tweepy
import redis
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

queue = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = queue.pubsub()
pip = queue.pipeline()


class MyListener(StreamListener):
    def on_data(self, data):
        try:
            queue.publish("tweet", data)
            #print(data)

        except BaseException as e:
            print("Publish error")
            return True


    def on_error(self, status):
        print(status)
        return True


usa=[-125.0011, 24.9493, -66.9326, 49.5904]
GEOBOX_GERMANY = [5.0770049095, 47.2982950435, 15.0403900146, 54.9039819757]
CA=[-123.970714,32.454870, -115.291514,42.017280]
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[ 'Traffic','TRAFFIC','CRASH','Crash','Accident','ACCIDENT','Stopped','Blocked','Road','Highway',
                             'hwy ','highway','hazard ','rd', 'latraffic','drive','ave', 'interstate','I-', 'incident',
                             'trafficsucks','trafficjam','trafficsuck','crash on','sactraffic','sactraffick',
                             'trfc' ,'collision' ,'st' ,'bridge','shoulder blocked',
                             'all lanes','backup','KCBSTraffic','lanes','traffic hazard','traffic collision','roadway','sdtraffic',
                             'right shoulder','left shoulder','right lane','left lane','lane blocked',
                             'bumper-to-bumper','bumper to bumper','bottleneck','jampacked','jam packed','jam-packed','stopped',
                             'congestion','gridlock','hold-up','holdup','hold up','traffic jam','accident','pile-up','pile up',
                             'pileup','traffic snarl','tail back','tailback','tail-back','nose to tail','nose-to-tail','tailback',
                             'heavy traffic','traffic block','traffic queue','snailpace','snail pace','slow traffic','traffic rerouted',
                             'traffic re-routed','road construction','terrible traffic','avoid road','massive traffic',
                             'traffic backed-up','traffic backed up','roundabout','vehicles not moving','fire','police activities',
                             'stuk up','stuck up','stuck-up','stuk-up','car crash','carcrash','traffic',
                             'tafficjams','sanjosetraffic','losangelestraffic','sandiegotraffic','wokvtraffic',
                             'sacramentotraffic','lbtraffic','longbeachtraffic','fog', 'heavy rain','hazard',
                             'intersection' ,'traffic signal' ,'lane','blvd','hazard road', 'avoid street','traffic queue'
                             #,'streetop','street','south','north','east','west','road','avoid street','streetopped'
    ],languages=["en"] ,locations=CA)
#country	    	longmin	latmin	longmax	latmax
#Germany	       	5.867	45.967	15.033	55.133
#United State		167.74	8.732	167.74	8.732

